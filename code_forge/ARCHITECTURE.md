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
- `digester/schema.py`
  - Strict artifact schema validation and contract checks.
- `digester/drift.py`
  - Run-over-run drift snapshots and regression warning reports.
- `bench/runner.py`
  - Repeatable ingestion/search/graph benchmark suite with regression gates.
- `canonicalize/planner.py`
  - Canonical migration map generation and compatibility shim staging.
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
Typed edges for structural and semantic traversal:
- `contains`
- `imports`
- `calls`
- `uses`

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
- emit `triage.json`, `triage.csv`, `triage_report.md`, `triage_audit.json`

### Stage D: Integration
- optional Knowledge Forge sync
- optional GraphRAG corpus export
- dependency graph export (`dependency_graph.json`)

### Stage E: Canonicalization Planning
- consume triage outputs
- build migration map from source -> canonical targets
- generate optional staged compatibility shims

### Stage F: Drift Intelligence
- compare current digester metrics vs previous snapshot
- emit `drift_report.json` + `drift_report.md`
- persist immutable snapshot into `history/*.json` for future comparisons

## Benchmark and Regression Gates

`bench/runner.py` produces measurable artifacts for:
- ingestion throughput (`files/s`, `units/s`)
- semantic search latency (`mean`, `p50`, `p95`, `max`)
- dependency graph build latency

When a baseline exists, gate checks fail on configured regressions.

## Safety and Idempotency

- `file_records` + `ANALYSIS_VERSION` prevent unnecessary reprocessing.
- Ingestion is non-destructive by default (`analysis` mode).
- Generated digester outputs are excluded from default ingestion scans.
- All stage outputs are persisted as explicit artifacts for review and rollback.
