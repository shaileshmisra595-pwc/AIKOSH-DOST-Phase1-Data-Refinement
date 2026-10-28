# AIKOSH DOST Phase 1

## Scope
Text/document unstructured data refinement into governed structured datasets.

## Pipeline
1. Data ingestion and inventory
2. Modality detection and routing
3. Text/document extraction
4. AI-assisted information-preserving Markdown normalization
5. Structure-aware chunking
6. ChromaDB persistence and lightweight retrieval
7. Structured JSON extraction with evidence and provenance
8. Deterministic normalization and duplicate detection
9. Quality validation: completeness, consistency, evidence coverage, PII indicators, schema health
10. Release artifacts: Markdown, JSON, CSV, quality report, provenance

## Operating modes
- RAG mode: retrieve relevant normalized chunks before extraction.
- Direct mode: extract from every normalized chunk without retrieval.

## Important distinction
The quality score in this application is an internal readiness score. It is not an official public benchmark score. Public benchmark claims require the benchmark dataset, ground truth and official evaluation protocol.
