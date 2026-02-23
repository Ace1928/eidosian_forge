# Code Forge

Code Forge is the code digestion and structural indexing subsystem of Eidosian Forge.
It ingests multi-language code, deduplicates and fingerprints units, builds searchable indexes,
and produces explainable triage outputs for archive reduction and canonical extraction.

## What It Does

- Multi-language ingestion into a normalized SQLite library (`code_units`, `code_text`, `relationships`, fingerprints, search index).
- Exact duplicate detection, normalized duplicate detection, and near-duplicate detection via SimHash.
- Structural clone clustering using identifier-abstracted structural hashes.
- Hybrid semantic search (FTS when available + lexical scoring fallback).
- Structural tracing of module/class/function containment graphs.
- Relationship edge extraction for `imports`, `calls`, and `uses`.
- Aggregated module dependency graph artifact generation (`dependency_graph.json`).
- Archive digester pipeline:
  - Stage A: intake catalog (`repo_index.json`)
  - Stage B: duplication index (`duplication_index.json`)
- Stage C: triage classification (`triage.json`, `triage.csv`, `triage_report.md`)
- Stage C.1: triage audit (`triage_audit.json`) with rule ids and confidence scores
- Stage D: dependency graph export (`dependency_graph.json`)
- Stage E: drift intelligence (`drift_report.json`, `history/*.json`) for run-over-run regression visibility
- Benchmark + regression gate suite for ingestion, semantic search, and dependency graph build latency.
- Eval/observability operating system:
  - declarative TaskBank contracts,
  - ablation config matrix execution,
  - replayable run artifacts (trace JSONL + stdout/stderr hashes + repo snapshot),
  - staleness/freshness metrics for memory-backed workflows.
- Canonical extraction planning with migration map and compatibility shim staging.
- Roundtrip reconstruction pipeline:
  - reconstruct source trees from library file records/blobs,
  - parity reports with hash-level verification,
  - transactional apply with backups and audit reports.
- Strict artifact schema validation (`validate-artifacts`) and strict digest failure mode.
- Integration exports:
  - Knowledge Forge sync (`sync-knowledge`)
  - Memory Forge sync (`sync-memory`)
  - GraphRAG corpus export (`export-graphrag`)

## Key Commands

```bash
# Health + index stats
code-forge status

# Ingest repository code (multi-language defaults)
code-forge ingest-dir . --mode analysis

# Duplicate and near-duplicate analysis
code-forge dedup-report
code-forge normalized-dedup-report
code-forge near-dedup-report --max-hamming 6 --min-tokens 20

# Hybrid semantic search and structural trace
code-forge semantic-search "workspace competition winner"
code-forge trace agent_forge.consciousness.kernel.ConsciousnessKernel --depth 3

# Build intake artifacts only
code-forge catalog . --output-dir data/code_forge/digester/latest

# Build dependency graph from relationship edges
code-forge dependency-graph --output-dir data/code_forge/digester/latest

# Validate artifact contracts
code-forge validate-artifacts --output-dir data/code_forge/digester/latest

# Validate roundtrip contracts (manifest/parity/summary/apply report)
code-forge validate-roundtrip \
  --workspace-dir data/code_forge/roundtrip/audit_forge \
  --verify-hashes

# Generate triage from existing intake artifacts
code-forge triage-report --output-dir data/code_forge/digester/latest

# Full archive-digester run
code-forge digest . \
  --output-dir data/code_forge/digester/latest \
  --sync-knowledge \
  --sync-memory \
  --export-graphrag \
  --integration-policy effective_run

# Generate drift report explicitly (uses latest history snapshot by default)
code-forge drift-report --output-dir data/code_forge/digester/latest

# Integration exports
code-forge sync-knowledge --kb-path data/kb.json
code-forge export-graphrag --output-dir data/code_forge/graphrag_input

# Regression-gated benchmark suite
code-forge benchmark \
  --root . \
  --output reports/code_forge_benchmark_latest.json \
  --baseline reports/code_forge_benchmark_baseline.json

# Coverage gate for code_forge/src/code_forge (CI uses same threshold)
./eidosian_venv/bin/python -m pytest code_forge/tests \
  --cov=code_forge/src/code_forge \
  --cov-report=term-missing \
  --cov-fail-under=70 -q

# Create sample eval contracts and run matrix
code-forge eval-init \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json

# Run eval matrix with trace + replay artifacts
code-forge eval-run \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json \
  --output-dir reports/code_forge_eval \
  --repeats 2 \
  --max-parallel 2 \
  --replay-mode record

# Optional OTLP export wiring (trace spans to external observability backend)
code-forge eval-run \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json \
  --output-dir reports/code_forge_eval \
  --otlp-endpoint http://127.0.0.1:4318 \
  --otlp-service-name code_forge_eval \
  --otlp-header Authorization=Bearer-token

# Replay from a prior record run using a shared replay store path
code-forge eval-run \
  --taskbank config/eval/taskbank.json \
  --matrix config/eval/config_matrix.json \
  --output-dir reports/code_forge_eval_replay \
  --replay-mode replay \
  --replay-store reports/code_forge_eval/replay_store

# Compute staleness metrics from provenance/freshness logs
code-forge eval-staleness \
  --input reports/code_forge_eval/freshness.jsonl \
  --output reports/code_forge_eval/staleness_metrics.json

# Canonical migration map and shim staging artifacts
code-forge canonical-plan \
  --triage-path data/code_forge/digester/latest/triage.json \
  --output-dir data/code_forge/canonicalization/latest

# Reconstruct a forge from the library and validate parity
code-forge reconstruct-project \
  --source-root audit_forge \
  --output-dir data/code_forge/roundtrip/audit_forge/reconstructed
code-forge parity-report \
  --source-root audit_forge \
  --reconstructed-root data/code_forge/roundtrip/audit_forge/reconstructed \
  --report-path data/code_forge/roundtrip/audit_forge/parity_report.json

# Full roundtrip (digest + integrations + reconstruction + parity + optional apply)
code-forge roundtrip audit_forge \
  --workspace-dir data/code_forge/roundtrip/audit_forge \
  --sync-knowledge \
  --sync-memory \
  --export-graphrag \
  --integration-policy effective_run \
  --apply
```

## Python API

```python
from pathlib import Path

from code_forge import (
    CodeLibraryDB,
    IngestionRunner,
    run_archive_digester,
)

db = CodeLibraryDB(Path("data/code_forge/library.sqlite"))
runner = IngestionRunner(db=db, runs_dir=Path("data/code_forge/ingestion_runs"))

summary = run_archive_digester(
    root_path=Path("."),
    db=db,
    runner=runner,
    output_dir=Path("data/code_forge/digester/latest"),
    mode="analysis",
)
print(summary["triage_report_path"])
```

## Data Contracts

Primary DB tables:
- `code_text`: deduplicated source blobs by SHA256
- `code_units`: normalized module/class/function/method/etc units
- `relationships`: typed edges (`contains`, `imports`, `calls`, `uses`)
- `code_fingerprints`: normalized hash, simhash64, token_count
- `code_search`: semantic/lexical search text
- `ingestion_runs`: deterministic ingestion run metadata

Primary digester artifacts:
- `repo_index.json`: deterministic file-level intake index
- `duplication_index.json`: exact/normalized/structural/near duplication report
- `dependency_graph.json`: file/module dependency graph from imports/calls/uses edges
- `triage.json`: explainable classification with metrics and reasons
- `triage_audit.json`: per-file rule trace with confidence values
- `triage.csv`: tabular triage export
- `triage_report.md`: human review report
- `archive_digester_summary.json`: full run summary
- `drift_report.json`: run-over-run metric comparison and warning set
- `drift_report.md`: human-readable drift summary
- `history/*.json`: immutable per-run metric snapshots used for drift comparison

Roundtrip artifacts:
- `reconstruction_manifest.json`: file-level reconstruction manifest with content-hash verification
- `parity_report.json`: source vs reconstructed hash-level parity result
- `roundtrip_summary.json`: end-to-end digest+integration+reconstruction+apply summary
- `Backups/code_forge_roundtrip/<transaction_id>/apply_report.json`: transactional apply audit record
- `signature` envelope on roundtrip JSON artifacts: deterministic payload hash (`payload_sha256`) plus artifact digest (`sha256` or `hmac-sha256` when `EIDOS_CODE_FORGE_SIGNING_KEY` is set)
- `provenance_links.json`: cross-forge provenance links (artifact checksums + knowledge/memory/GraphRAG linkages)

Canonicalization artifacts:
- `migration_map.json`: source→canonical mapping with strategy labels
- `canonicalization_plan.md`: actionable migration plan
- `canonicalization_summary.json`: plan summary metadata
- `shims/**`: staged compatibility shim files (optional)

Eval/observability artifacts:
- `reports/code_forge_eval/summary.json`: top-level report (success rate, config scores, replay stats)
- `reports/code_forge_eval/runs/<run_id>/trace.jsonl`: append-only span/event trace
- `reports/code_forge_eval/runs/<run_id>/stdout.txt` and `stderr.txt`: artifact-backed command outputs
- `reports/code_forge_eval/replay_store/**`: deterministic record/replay command outputs keyed by task+config+command hash

## Integration Map

- `knowledge_forge`: `sync_units_to_knowledge_forge`
- `memory_forge`: `sync_units_to_memory_forge` (JSON episodic store)
- `graphrag_forge`: `export_units_for_graphrag`
- `scripts/living_knowledge_pipeline.py`: now includes Code Forge digester artifacts in code analysis report

## Planning + Evidence

- Roundtrip validation evidence (`audit_forge`): `code_forge/docs/ROUNDTRIP_AUDIT_FORGE_CYCLE_2026-02-19.md`
- Roundtrip validation evidence (`sms_forge`): `code_forge/docs/ROUNDTRIP_SMS_FORGE_CYCLE_2026-02-19.md`
- Next cycle implementation plan: `code_forge/docs/NEXT_CYCLE_PLAN_2026-02-19.md`
- Next cycle implementation plan (Cycle 07): `code_forge/docs/NEXT_CYCLE_PLAN_2026-02-20.md`
- Provenance schema/reference: `code_forge/docs/PROVENANCE_MODEL.md`
- Operator tutorial: `code_forge/docs/TUTORIAL_ROUNDTRIP_SAMPLE.md`

## Engineering Notes

- Designed for idempotent ingestion (`file_records` + `ANALYSIS_VERSION`).
- Integration exports are scoped to the active ingestion run when possible; if no new units are created, exports fall back to the latest effective run for that source root.
- Integration scope policy is configurable with `--integration-policy {run,effective_run,global}` on `digest` and `roundtrip`.
- Apply supports `--require-manifest` and `--dry-run` for guarded promotion workflows.
- Apply/replacement is managed-scope safe: prune/removal only applies to manifest-managed paths (or explicit scoped filters), preventing unmanaged file deletion.
- Roundtrip parity hashing can run in parallel for larger trees; tune with `EIDOS_CODE_FORGE_HASH_WORKERS` (defaults to auto-scaling for large file sets).
- FTS5 search is used when available; fallback lexical search remains active.
- Default ingestion excludes generated outputs (`data/code_forge/digester`, `data/code_forge/graphrag_input`, `doc_forge/final_docs`).
