import csv, json, math, os, re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .extractors import extract_text, basic_inventory, split_raw
from .sarvam import SarvamClient
from .rag import ChromaRAG

BASE_DIR=Path(__file__).resolve().parents[1]
OUTPUT_DIR=BASE_DIR/"outputs"

MARKDOWN_SYSTEM="""You are the information-preserving normalization engine for AIKOSH DOST Phase 1.
Convert the supplied source into clean Markdown for downstream retrieval and extraction.
STRICT SOURCE-FAITHFULNESS RULES:
1. Preserve every factual detail that is present in the source, including names, dates, numbers, units, headings, tables, lists, relationships and qualifiers.
2. Do not invent, infer, guess, correct, modernize, paraphrase away, or fill missing information.
3. Never introduce facts from your general knowledge.
4. If a value is unclear, preserve the uncertainty or write [UNREADABLE].
5. Do not merge separate facts unless the source itself connects them.
6. Preserve table structure as Markdown tables when the source supports it.
7. Return Markdown only.
"""

EXTRACT_SYSTEM="""You are the structured data extraction engine for AIKOSH DOST Phase 1.
You are given source context and a user-defined use case.
STRICT ANTI-HALLUCINATION RULES:
1. Extract ONLY information explicitly supported by the supplied source context.
2. Never use outside knowledge, web knowledge, assumptions, or likely values.
3. Never manufacture names, dates, amounts, entities, categories, relationships, or missing values.
4. If a requested field is absent, return null or an empty array according to its type.
5. Preserve source wording for factual values whenever practical.
6. Every record MUST contain an evidence field containing a concise verbatim fragment from the supplied source context that directly supports the record.
7. Evidence must come from the supplied source context. Never write invented evidence.
8. Keep the schema stable across chunks. Do not create a new synonym for an existing field merely because the wording changes.
9. Return valid JSON only.
10. Prefer one document-level record when the source is a single document. Create multiple records only when the source clearly contains multiple distinct entities/rows/events.
JSON shape: {"document_metadata":{},"schema":[{"field":"...","description":"...","data_type":"string|number|integer|boolean|array|object"}],"records":[{...}]}.
"""

def parse_json(s):
    s=s.strip(); s=re.sub(r"^```json\s*","",s,flags=re.I); s=re.sub(r"^```\s*","",s); s=re.sub(r"\s*```$","",s); return json.loads(s)

def normalize_records(records):
    out=[]; seen=set()
    for r in records:
        if not isinstance(r,dict): continue
        clean={}
        for k,v in r.items():
            clean[str(k).strip()]=re.sub(r"\s+"," ",v).strip() if isinstance(v,str) else v
        key=json.dumps({k:v for k,v in clean.items() if k!="evidence"},sort_keys=True,ensure_ascii=False,default=str)
        if key not in seen: seen.add(key); out.append(clean)
    return out

def _empty(v): return v is None or v=="" or v==[] or v=={}
def _type_ok(v,t):
    if _empty(v): return True
    t=(t or "string").lower()
    if t in {"string","text","date","datetime"}: return isinstance(v,str)
    if t in {"integer","int"}: return isinstance(v,int) and not isinstance(v,bool)
    if t in {"number","float","decimal"}: return isinstance(v,(int,float)) and not isinstance(v,bool)
    if t in {"boolean","bool"}: return isinstance(v,bool)
    if t in {"array","list"}: return isinstance(v,list)
    if t in {"object","json"}: return isinstance(v,dict)
    return True

def _norm(v): return re.sub(r"[^a-z0-9]+"," ",str(v).lower()).strip()
def _tokens(v):
    if isinstance(v,list): return sum((_tokens(x) for x in v),[])
    if isinstance(v,dict): return sum((_tokens(x) for x in v.values()),[])
    return [x for x in _norm(v).split() if len(x)>1]

def _parse_date(v):
    if not isinstance(v,str): return None
    for fmt in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%m/%d/%Y","%Y/%m/%d","%B %d, %Y","%d %B %Y"):
        try: return datetime.strptime(v.strip(),fmt).date()
        except ValueError: pass
    return None

def chunk_quality(chunks, max_chars, overlap):
    if not chunks: return {"score":0,"checks":[],"status":"FAIL"}
    lengths=[len(x) for x in chunks]
    empty=sum(1 for x in chunks if not x.strip())
    oversized=sum(1 for x in lengths if x>max_chars)
    boundary_ok=sum(1 for x in chunks if x.strip() and not x.strip().endswith("[TRUNCATED]"))
    overlaps=[]
    for a,b in zip(chunks,chunks[1:]):
        suffix=a[-overlap:] if overlap else ""
        overlaps.append(1 if suffix and b.startswith(suffix) else 0)
    checks=[
      {"dimension":"Non-empty chunks","score":round(100*(1-empty/len(chunks)),2),"detail":{"empty_chunks":empty}},
      {"dimension":"Size compliance","score":round(100*(1-oversized/len(chunks)),2),"detail":{"max_allowed":max_chars,"oversized_chunks":oversized,"min":min(lengths),"max":max(lengths),"average":round(sum(lengths)/len(lengths),2)}},
      {"dimension":"Boundary integrity","score":round(100*boundary_ok/len(chunks),2),"detail":{"clean_boundaries":boundary_ok,"total":len(chunks)}},
      {"dimension":"Overlap continuity","score":round(100*sum(overlaps)/len(overlaps),2) if overlaps else 100.0,"detail":{"configured_overlap":overlap,"verified_pairs":sum(overlaps),"pairs":len(overlaps)}},
    ]
    score=round(sum(c["score"] for c in checks)/len(checks),2)
    return {"score":score,"status":"PASS" if score>=95 else "REVIEW" if score>=80 else "FAIL","checks":checks,"chunk_count":len(chunks)}

def embedding_quality(embedding_meta,retrieval):
    rows=embedding_meta.get("preview",[]); dims=embedding_meta.get("dimensions",0)
    finite=sum(1 for r in rows if all(math.isfinite(float(x)) for x in r.get("first_12_dimensions",[])))
    nonzero=sum(1 for r in rows if r.get("norm",0)>0)
    dim_ok=sum(1 for r in rows if r.get("dimensions")==dims)
    ids=[r.get("chunk_id") for r in rows]; unique=len(ids)==len(set(ids))
    retrieval_ok=sum(1 for r in retrieval if r.get("hits"))
    n=max(1,len(rows))
    checks=[
      {"dimension":"Dimension consistency","score":round(100*dim_ok/n,2),"detail":{"expected_dimensions":dims}},
      {"dimension":"Non-zero vectors","score":round(100*nonzero/n,2),"detail":{"zero_vectors":n-nonzero}},
      {"dimension":"Finite vector values","score":round(100*finite/n,2),"detail":{"invalid_preview_vectors":n-finite}},
      {"dimension":"Unique chunk IDs","score":100.0 if unique else 0.0,"detail":{"unique":unique}},
      {"dimension":"Retrieval coverage","score":round(100*retrieval_ok/max(1,len(retrieval)),2),"detail":{"queries_with_hits":retrieval_ok,"queries":len(retrieval)}},
    ]
    score=round(sum(c["score"] for c in checks)/len(checks),2)
    return {"score":score,"status":"PASS" if score>=95 else "REVIEW" if score>=80 else "FAIL","checks":checks,"note":"These checks validate vector/index integrity and retrieval behavior. They do not prove semantic correctness of an embedding model."}

def six_quality_checks(records,schema,source):
    fields=[s.get("field") for s in schema if s.get("field")]; data=[f for f in fields if f!="evidence"]
    meta_fields={"title","headline","source","publication","reporting_source","date","author","document_type"}
    applicable=[f for f in data if f not in meta_fields] or data
    applicable_cells=len(records)*len(applicable); missing=sum(1 for r in records for f in applicable if _empty(r.get(f)))
    completeness=round(100*(1-missing/max(1,applicable_cells)),2) if records else 0
    sig=[json.dumps({k:v for k,v in r.items() if k!="evidence"},sort_keys=True,ensure_ascii=False,default=str) for r in records]
    dup=len(sig)-len(set(sig)); uniqueness=round(100*(1-dup/max(1,len(records))),2) if records else 0
    type_checks=type_pass=0; vf=[]; smap={s.get("field"):s for s in schema}
    for i,r in enumerate(records,1):
        for f in data:
            if _empty(r.get(f)): continue
            type_checks+=1
            if _type_ok(r[f],smap.get(f,{}).get("data_type")): type_pass+=1
            else: vf.append({"record":i,"field":f,"reason":"Declared data type mismatch"})
    validity=round(100*type_pass/max(1,type_checks),2) if records else 0
    consistency=100.0; cf=[]
    entity_values=defaultdict(lambda:defaultdict(set))
    for r in records:
        entity=r.get("entity") or r.get("name") or r.get("company") or r.get("subject_entity") or r.get("title")
        ek=_norm(entity)
        if not ek: continue
        for f,v in r.items():
            if f=="evidence" or _empty(v): continue
            entity_values[ek][f].add(json.dumps(v,sort_keys=True,ensure_ascii=False,default=str))
    total_slots=sum(len(v) for v in entity_values.values()); contradictions=0
    for ek,fm in entity_values.items():
        for f,vals in fm.items():
            if len(vals)>1: contradictions+=1; cf.append({"entity":ek,"field":f,"distinct_values":len(vals)})
    if total_slots: consistency=round(100*(1-contradictions/total_slots),2)
    evidence_supported=evidence_applicable=0; af=[]
    for i,r in enumerate(records,1):
        ev=_norm(r.get("evidence",""))
        for f in data:
            v=r.get(f)
            if _empty(v): continue
            evidence_applicable+=1; ts=_tokens(v)
            matched=sum(1 for t in ts if t in ev)
            if ts and (" ".join(ts) in ev or matched>=max(1,math.ceil(len(ts)*.6))): evidence_supported+=1
            else: af.append({"record":i,"field":f,"reason":"Value lacks sufficient evidence support"})
    accuracy=round(100*evidence_supported/max(1,evidence_applicable),2) if records else 0
    date_fields=[f for f in fields if f.lower() in {"date","publication_date","published_date","updated_at","timestamp","datetime"}]
    tv=[(i,r.get(f),f) for i,r in enumerate(records,1) for f in date_fields if not _empty(r.get(f))]
    tf=[]
    if tv:
        today=datetime.now(timezone.utc).date(); valid=sum(1 for i,v,f in tv if _parse_date(v) and _parse_date(v)<=today)
        timeliness=round(100*valid/len(tv),2)
        tf=[{"record":i,"field":f,"reason":"Missing, unparseable, or future date"} for i,v,f in tv if not (_parse_date(v) and _parse_date(v)<=today)]
        tapp=True
    else: timeliness=None; tapp=False
    checks=[
      {"dimension":"Completeness","score":completeness,"status":"PASS" if completeness>=95 else "REVIEW" if completeness>=80 else "FAIL","applicable":True,"findings":[{"missing_cells":missing,"applicable_cells":applicable_cells}]},
      {"dimension":"Uniqueness","score":uniqueness,"status":"PASS" if uniqueness>=98 else "REVIEW" if uniqueness>=90 else "FAIL","applicable":True,"findings":[{"duplicate_records":dup}]},
      {"dimension":"Validity","score":validity,"status":"PASS" if validity>=95 else "REVIEW" if validity>=80 else "FAIL","applicable":True,"findings":vf[:50]},
      {"dimension":"Consistency","score":consistency,"status":"PASS" if consistency>=95 else "REVIEW" if consistency>=80 else "FAIL","applicable":True,"findings":cf[:50]},
      {"dimension":"Accuracy","score":accuracy,"status":"PASS" if accuracy>=95 else "REVIEW" if accuracy>=80 else "FAIL","applicable":True,"findings":af[:50]},
      {"dimension":"Timeliness","score":timeliness,"status":"N/A" if not tapp else "PASS" if timeliness>=95 else "REVIEW" if timeliness>=80 else "FAIL","applicable":tapp,"findings":tf[:50]},
    ]
    scores=[c["score"] for c in checks if c["applicable"] and c["score"] is not None]; overall=round(sum(scores)/len(scores),2) if scores else 0
    verdict="NOT FIT" if any(c["status"]=="FAIL" for c in checks if c["applicable"]) or overall<65 else "CONDITIONAL" if any(c["status"]=="REVIEW" for c in checks if c["applicable"]) or overall<85 else "FIT"
    emails=len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",source)); phones=len(re.findall(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",source)); aadhaar=len(re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b",source)); pan=len(re.findall(r"\b[A-Z]{5}\d{4}[A-Z]\b",source,re.I)); pii=emails+phones+aadhaar+pan
    return {"score":overall,"verdict":verdict,"metrics":{"records":len(records),"fields":len(fields),"completeness_percent":completeness,"evidence_coverage_percent":round(100*sum(1 for r in records if r.get("evidence"))/max(1,len(records)),2),"pii_indicators":pii},"six_data_quality_checks":checks,"privacy":{"pii_detected":pii>0,"findings":{"email":emails,"phone":phones,"aadhaar_like":aadhaar,"pan_like":pan}},"diagnostics":{"applicable_cells":applicable_cells,"populated_cells":applicable_cells-missing,"unsupported_missing_cells":missing,"metadata_nulls_not_penalized":True,"accuracy_is_evidence_support_proxy":True}}

def flatten(r):
    return {k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()}

def run_pipeline(path,run_id,mode="rag",use_case=""): 
    out=OUTPUT_DIR/run_id; out.mkdir(parents=True,exist_ok=True)
    max_source=int(os.getenv("MAX_SOURCE_CHARS","900000")); max_chars=int(os.getenv("CHUNK_CHARS","9000")); overlap=int(os.getenv("CHUNK_OVERLAP","700")); top_k=int(os.getenv("RAG_TOP_K","5"))
    raw,inventory=extract_text(path)
    modality=inventory.copy(); modality["modality"] = inventory.get("modality", "document")
    ocr_info=None
    if inventory.get("needs_ocr") or inventory.get("scanned_pages"):
        client=SarvamClient(); raw,ocr_info=client.document_markdown(path,os.getenv("SARVAM_LANGUAGE","en-IN")); inventory["ocr_provider"]="Sarvam Document AI"; inventory["ocr_status"]=ocr_info.get("status"); inventory["ocr_job_id"]=ocr_info.get("job_id"); inventory["ocr_applied"]=True
    else: inventory["ocr_applied"]=False
    if len(raw)>max_source: raise ValueError(f"Source is {len(raw):,} characters. Configured maximum is {max_source:,}.")
    inventory.update(basic_inventory(path,raw)); inventory["modality"]=modality.get("modality","document")
    if not raw.strip(): raise ValueError("No extractable source content was found. For scanned/image documents, enable Sarvam Document AI OCR.")
    client=SarvamClient()
    raw_chunks=split_raw(raw,max_chars,overlap)
    markdown_chunks=[]; usage=[]
    for i,c in enumerate(raw_chunks):
        res=client.chat([{"role":"system","content":MARKDOWN_SYSTEM},{"role":"user","content":f"SOURCE CHUNK chunk-{i+1:04d}\n\n{c}"}],max_tokens=7000,temperature=0)
        markdown_chunks.append(res["content"].strip()); usage.append(res.get("usage",{}))
    markdown="\n\n".join(f"## Source Chunk {i+1:04d}\n\n{c}" for i,c in enumerate(markdown_chunks)); (out/"normalized.md").write_text(markdown,encoding="utf-8")
    md_chunks=split_raw(markdown,max_chars,overlap); cq=chunk_quality(md_chunks,max_chars,overlap)
    rag=ChromaRAG(OUTPUT_DIR,run_id); emb=rag.index(md_chunks); retrieval=[]; extracted=[]
    for i,c in enumerate(md_chunks):
        hits=rag.retrieve(c,k=top_k) if mode=="rag" else []
        retrieval.append({"query_chunk":f"chunk-{i+1:04d}","hits":[{"chunk_id":h["chunk_id"],"distance":h["distance"]} for h in hits]})
        context="\n\n".join(f"SOURCE {h['chunk_id']}:\n{h['text']}" for h in hits) if hits else c
        prompt=f"USE CASE:\n{use_case.strip() or 'General document structuring'}\n\nTARGET CHUNK: chunk-{i+1:04d}\n\nSOURCE CONTEXT:\n{context}\n\nExtract only source-supported information."
        res=client.json_chat(EXTRACT_SYSTEM,prompt,None,max_tokens=7000); extracted.append(parse_json(res["content"]))
    eq=embedding_quality(emb,retrieval)
    schema_map={}; records=[]; metadata={}
    for item in extracted:
        for f in item.get("schema",[]):
            name=str(f.get("field","")).strip()
            if name and name not in schema_map: schema_map[name]=f
        if isinstance(item.get("document_metadata"),dict): metadata.update(item["document_metadata"])
        if isinstance(item.get("records"),list): records.extend(item["records"])
    records=normalize_records(records); schema=list(schema_map.values())
    if "evidence" not in [s.get("field") for s in schema]: schema.append({"field":"evidence","description":"Concise verbatim source fragment supporting the extracted record","data_type":"string"})
    fields=[s["field"] for s in schema if s.get("field")]
    with (out/"structured.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); [w.writerow(flatten(r)) for r in records]
    structured={"document_metadata":metadata,"schema":schema,"records":records}; (out/"structured.json").write_text(json.dumps(structured,ensure_ascii=False,indent=2),encoding="utf-8")
    from openpyxl import Workbook
    wb=Workbook(); ws=wb.active; ws.title="Dataset"; ws.append(fields)
    for r in records: ws.append([flatten(r).get(f) for f in fields])
    ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions; wb.save(out/"structured.xlsx")
    chunks_payload=[{"chunk_id":f"chunk-{i+1:04d}","characters":len(c),"text":c} for i,c in enumerate(md_chunks)]; (out/"chunks.json").write_text(json.dumps(chunks_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    quality=six_quality_checks(records,schema,raw); quality["framework"]={"name":"AIKOSH DOST Deliverable 3","phase":"Phase 1","scope":"Text/document unstructured data refinement","score_type":"Internal readiness score","mode":mode,"benchmark_claim":False}; quality["extraction_diagnostics"]={"records_extracted":len(records),"schema_fields":len(fields),"document_level_extraction":True,"sparse_record_merge":False}
    (out/"quality_report.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding="utf-8")
    emb_preview={"model":emb.get("model"),"method":emb.get("method"),"dimensions":emb.get("dimensions"),"chunks":emb.get("preview",[]),"quality":eq}; (out/"embedding_preview.json").write_text(json.dumps(emb_preview,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"chunk_quality.json").write_text(json.dumps(cq,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"retrieval.json").write_text(json.dumps(retrieval,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"source_preview.txt").write_text(raw[:12000],encoding="utf-8")
    provenance={"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"source":inventory,"use_case":use_case,"model":client.model,"mode":mode,"stages":["modality_check","source_extraction","sarvam_document_ocr_if_required","sarvam_markdown_normalization","user_use_case","controlled_chunking","chunk_quality_checks","vector_embedding","embedding_quality_checks","retrieval","sarvam_structured_extraction","six_dimension_data_quality","artifact_release"],"raw_chunk_count":len(raw_chunks),"normalized_chunk_count":len(md_chunks),"model_usage":usage,"ocr":ocr_info}
    (out/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    return {
        "run_id": run_id,
        "inventory": inventory,
        "use_case": use_case,
        "mode": mode,
        "stage_summary": {"source_chunks": len(raw_chunks), "normalized_chunks": len(md_chunks), "records": len(records), "fields": len(fields), "embedded_chunks": len(md_chunks)},
        "previews": {
            "source": raw[:12000],
            "modality": {"extension": path.suffix.lower(), "detected_modality": inventory.get("modality"), "ocr_applied": inventory.get("ocr_applied"), "ocr": ocr_info},
            "markdown": markdown[:12000],
            "use_case": use_case,
            "chunks": [{"chunk_id": f"chunk-{i+1:04d}", "characters": len(c), "preview": c[:1800]} for i,c in enumerate(md_chunks[:20])],
            "chunk_quality": cq,
            "embedding": emb_preview,
            "embedding_quality": eq,
            "json": structured,
            "table": [flatten(r) for r in records[:50]],
        },
        "structured": {"schema": schema, "record_count": len(records), "preview": records[:20]},
        "retrieval": retrieval[:20],
        "quality": quality,
        "artifacts": {
            "source": f"/api/download/{run_id}/source",
            "markdown": f"/api/download/{run_id}/markdown",
            "chunks": f"/api/download/{run_id}/chunks",
            "chunk_quality": f"/api/download/{run_id}/chunk-quality",
            "embedding": f"/api/download/{run_id}/embedding",
            "json": f"/api/download/{run_id}/json",
            "csv": f"/api/download/{run_id}/csv",
            "xlsx": f"/api/download/{run_id}/xlsx",
            "report": f"/api/download/{run_id}/report",
            "provenance": f"/api/download/{run_id}/provenance",
            "retrieval": f"/api/download/{run_id}/retrieval",
        },
    }
