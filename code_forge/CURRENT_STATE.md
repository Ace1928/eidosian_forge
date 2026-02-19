# Current State: Code Forge

**Date**: 2026-02-19  
**Status**: Production active, archive-digester foundation implemented  
**Version**: 1.0.x

## Snapshot

- Multi-language ingestion (Python + broad generic language coverage by extension).
- SQLite-backed code library with:
  - normalized units
  - deduplicated source blobs
  - relationship edges
  - fingerprint/search indexes
- Duplicate analysis surfaces:
  - exact duplicate groups
  - normalized duplicate groups
  - structural duplicate groups
  - near-duplicate pairs (SimHash/Hamming)
- Hybrid semantic search (FTS5 when available, lexical fallback otherwise).
- Relationship extraction and graphing:
  - `imports`, `calls`, `uses` edge ingestion
  - `dependency_graph.json` artifact generation
- Archive digester artifacts:
  - `repo_index.json`
  - `duplication_index.json`
  - `triage.json`
  - `triage.csv`
  - `triage_report.md`
  - `drift_report.json`
  - `history/*.json` snapshots
  - `archive_digester_summary.json`
- Roundtrip/regeneration artifacts:
  - `reconstruction_manifest.json`
  - `parity_report.json`
  - `roundtrip_summary.json`
  - transactional backup/apply reports under `Backups/code_forge_roundtrip/*`

## Validation

- Test suite expanded and passing under `eidosian_venv` (`code_forge/tests`).
- Benchmark suite implemented with baseline-aware regression gates.
- Canonical migration planner implemented with staged compatibility shim generation.
- Triage now emits ruleset-bound confidence and `triage_audit.json`.
- Digester artifacts validated against schema contracts (`validate-artifacts` CLI).
- Digester now emits drift reports and persistent history snapshots for run-over-run monitoring.
- Ingestion remains idempotent via `file_records` and `ANALYSIS_VERSION` gates.
- Living knowledge pipeline now emits richer code analysis outputs (language split, triage references).
- Roundtrip CLI flow implemented (`reconstruct-project`, `parity-report`, `apply-reconstruction`, `roundtrip`).
- Integration exports now scope to the active run and fall back to latest effective source-root run if no new units were produced.

## Open Gaps

- Signed/tamper-evident artifact manifests are not implemented yet.
- Large-tree (>10k files) parallel parity hashing and regeneration stress baselines are pending.
- Root-scoped export policy modes (`run` vs `effective_run` vs `global`) are not user-configurable yet.
