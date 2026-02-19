# Code Forge

Code Forge is the code digestion and structural indexing subsystem of Eidosian Forge.
It ingests multi-language code, deduplicates and fingerprints units, builds searchable indexes,
and produces explainable triage outputs for archive reduction and canonical extraction.

## What It Does

- Multi-language ingestion into a normalized SQLite library (`code_units`, `code_text`, `relationships`, fingerprints, search index).
- Exact duplicate detection, normalized duplicate detection, and near-duplicate detection via SimHash.
- Hybrid semantic search (FTS when available + lexical scoring fallback).
- Structural tracing of module/class/function containment graphs.
- Archive digester pipeline:
  - Stage A: intake catalog (`repo_index.json`)
  - Stage B: duplication index (`duplication_index.json`)
  - Stage C: triage classification (`triage.json`, `triage.csv`, `triage_report.md`)
- Integration exports:
  - Knowledge Forge sync (`sync-knowledge`)
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

# Generate triage from existing intake artifacts
code-forge triage-report --output-dir data/code_forge/digester/latest

# Full archive-digester run
code-forge digest . \
  --output-dir data/code_forge/digester/latest \
  --sync-knowledge \
  --export-graphrag

# Integration exports
code-forge sync-knowledge --kb-path data/kb.json
code-forge export-graphrag --output-dir data/code_forge/graphrag_input
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
- `relationships`: typed edges (currently `contains`)
- `code_fingerprints`: normalized hash, simhash64, token_count
- `code_search`: semantic/lexical search text
- `ingestion_runs`: deterministic ingestion run metadata

Primary digester artifacts:
- `repo_index.json`: deterministic file-level intake index
- `duplication_index.json`: exact/normalized/near duplication report
- `triage.json`: explainable classification with metrics and reasons
- `triage.csv`: tabular triage export
- `triage_report.md`: human review report
- `archive_digester_summary.json`: full run summary

## Integration Map

- `knowledge_forge`: `sync_units_to_knowledge_forge`
- `graphrag_forge`: `export_units_for_graphrag`
- `scripts/living_knowledge_pipeline.py`: now includes Code Forge digester artifacts in code analysis report

## Engineering Notes

- Designed for idempotent ingestion (`file_records` + `ANALYSIS_VERSION`).
- FTS5 search is used when available; fallback lexical search remains active.
- Default ingestion excludes generated outputs (`data/code_forge/digester`, `data/code_forge/graphrag_input`, `doc_forge/final_docs`).
