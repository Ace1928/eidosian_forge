# Code Forge Architecture

## Purpose
Code Forge is the code substrate for Eidosian Forge: it turns heterogeneous source files into normalized, searchable, and triageable code intelligence artifacts.

## Runtime Components

- `analyzer/python_analyzer.py`
  - AST-accurate Python extraction (module/class/function/method/node granularity).
- `analyzer/generic_analyzer.py`
  - Regex-backed multi-language fallback analyzer.
- `ingest/runner.py`
  - Idempotent ingestion runner with run manifests.
- `library/db.py`
  - SQLite schema for units, text blobs, relationships, fingerprints, and search index.
- `library/similarity.py`
  - Normalization/tokenization/fingerprint primitives (`normalized_hash`, `simhash64`).
- `digester/pipeline.py`
  - Archive Digester orchestration (intake -> dedup -> triage -> integration exports).
- `integration/pipeline.py`
  - Knowledge Forge sync and GraphRAG corpus export.
- `cli.py`
  - Operator surface for ingestion, search, dedup, triage, and digestion workflows.

## Core Data Model

### `code_units`
Canonical unit records (module/class/function/method/etc):
- identity: `id`, `language`, `unit_type`, `qualified_name`
- location: `file_path`, line/column span
- content linkage: `content_hash`
- run linkage: `run_id`
- quality fields: `complexity`

### `code_fingerprints`
Per-unit dedup/search features:
- `normalized_hash`
- `simhash64`
- `token_count`

### `code_search`
Unit-level search corpus used by:
- FTS5 query path (when available)
- lexical fallback path

### `relationships`
Typed edges (`contains` today) for structural traversal and trace output.

## Archive Digester Stages

### Stage A: Intake
- scan and hash files
- emit deterministic `repo_index.json`

### Stage B: Duplication
- exact duplicate groups
- normalized duplicate groups
- near-duplicate pairs
- emit `duplication_index.json`

### Stage C: Triage
- aggregate file metrics from indexed units
- classify `keep` / `extract` / `refactor` / `quarantine` / `delete_candidate`
- emit `triage.json`, `triage.csv`, `triage_report.md`

### Stage D: Integration
- optional Knowledge Forge sync
- optional GraphRAG corpus export

## Safety and Idempotency

- `file_records` + `ANALYSIS_VERSION` prevent unnecessary reprocessing.
- Ingestion is non-destructive by default (`analysis` mode).
- Generated digester outputs are excluded from default ingestion scans.
- All stage outputs are persisted as explicit artifacts for review and rollback.
