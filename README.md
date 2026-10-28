# AIKOSH DOST Phase 1 Preview + Quality Workbench

## Intended workflow

Dataset → modality check → OCR/text extraction → Markdown → user use case → chunking → chunk quality → vector embedding/index → embedding quality → retrieval → structured JSON/CSV/XLSX → six data-quality checks → release.

Every major transformation has a UI preview and a downloadable artifact.

## Sarvam

The application uses Sarvam Chat Completions with `sarvam-105b` for Markdown normalization and source-grounded structured extraction. For scanned PDFs and image inputs it can use Sarvam Document AI Parse to produce Markdown before downstream processing. Sarvam documents its Parse API as full-document OCR with Markdown/HTML/JSON output and its Chat API supports strict JSON Schema structured output.

Never commit your real API key. Put it in `.env` locally.

## Use case

The UI requires a use-case description before a run. It is passed to the extraction prompt so the schema and fields serve the stated downstream purpose. The prompt explicitly forbids outside knowledge, guessing, invented evidence, and unsupported values.

## Quality design

### Dataset quality
Six dimensions are applied after structured extraction:

1. Completeness
2. Uniqueness
3. Validity
4. Consistency
5. Accuracy, implemented as deterministic evidence-support coverage
6. Timeliness

Accuracy is not presented as external truth verification. It checks whether extracted values are supported by the supplied source evidence.

### Chunk quality
Chunk checks cover non-empty chunks, size compliance, boundary integrity and overlap continuity.

### Vector quality
Vector checks cover dimension consistency, non-zero vectors, finite values, unique chunk IDs and retrieval coverage. These checks validate index integrity and retrieval behavior. They do not prove that an embedding model is semantically correct.

For stronger RAG evaluation, the project UI links to DeepEval, Ragas and Arize Phoenix. DeepEval provides retriever metrics such as contextual relevancy, precision and recall, plus generator metrics such as faithfulness and answer relevancy. Reference-based metrics require expected answers, while referenceless metrics can be used without labels.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and add your Sarvam key
python app.py
```

Open `http://127.0.0.1:8000`.

## Large datasets

The current UI is a document-level refinement workbench. For very large collections, process files individually or add a batch queue in the next phase. Sarvam Document AI currently documents a 30-page PDF limit for Parse and a 20-page PDF limit for Extract, so large PDFs should be split before those operations.
