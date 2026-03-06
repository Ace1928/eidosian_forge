# Living Knowledge System

## Goal

Create a strict, continuously refreshable knowledge substrate that unifies:

- memory artifacts (`memory_data.json`, tiered memory)
- knowledge graph artifacts (`data/kb.json`)
- repository documentation
- repository code and code-structure traces

Then stage that unified corpus into GraphRAG for relationship-centric querying and evaluation.

## Why This Design

- GraphRAG is built for graph-centered retrieval and global/local query modes, which fits entity/relation-heavy operational memory and codebase reasoning.
- A single repeatable contract (`manifest.json` + `records.jsonl` + dedup/drift reports) prevents "silent drift" and supports objective comparisons run-to-run.
- Code-level trace + dedup reports from `code_forge` provide structural constraints before semantic indexing.

## Pipeline Entry Point

- Script: `scripts/living_knowledge_pipeline.py`
- Primary output root: `reports/living_knowledge/<run_id>/`
- Optional GraphRAG workspace: `data/living_knowledge/workspace/`

## Output Contract (`living_knowledge.pipeline.v1`)

Per run:

- `manifest.json`
  - run metadata
  - record counts by kind
  - dedup counts
  - drift summary against previous run
  - code analysis summary
  - GraphRAG execution summary
- `records.jsonl`
  - one staged document per line with deterministic identifiers
- `duplicates_exact.json`
  - exact duplicate groups by SHA-256
- `duplicates_near.json`
  - near duplicates by SimHash/Hamming threshold
- `drift.json`
  - added/removed/changed source paths vs previous run
- `code_analysis_report.json`
  - code ingestion stats, unit counts, duplicate code unit groups, trace samples

## Code Forge Integration

`scripts/living_knowledge_pipeline.py` executes code ingestion using:

- `code_forge.ingest.runner.IngestionRunner`
- `code_forge.library.db.CodeLibraryDB`

and persists:

- duplicate code groups (`content_hash` collisions across units)
- sampled contains-graph traces
- ingestion run metrics

New CLI capabilities for targeted workflows:

- `code-forge dedup-report`
- `code-forge trace <qualified_name_or_unit_id>`

## GraphRAG Integration

When `--run-graphrag` is enabled:

- writes strict workspace settings and prompts to `data/living_knowledge/workspace`
- starts local completion + embedding llama servers
- runs `python -m graphrag index`
- optionally executes one or more global queries

Default completion and embedding models are selected from `config/model_selection.json`.
If no selection file exists, completion falls back to latest sweep winner (`reports/graphrag_sweep/model_selection_latest.json`) and then local Qwen 0.5B.

## Usage

```bash
./eidosian_venv/bin/python scripts/living_knowledge_pipeline.py \
  --repo-root . \
  --output-root reports/living_knowledge \
  --workspace-root data/living_knowledge/workspace \
  --code-max-files 2000

./eidosian_venv/bin/python scripts/living_knowledge_pipeline.py \
  --run-graphrag \
  --query "What duplicated logic patterns recur across forges?"
```

### Interop Validation (Knowledge/Memory/Code/Word + GraphRAG)

Use this validation matrix for production checks:

```bash
./eidosian_venv/bin/python -m pytest -q code_forge/tests
./eidosian_venv/bin/python -m pytest -q knowledge_forge/tests
./eidosian_venv/bin/python -m pytest -q memory_forge/tests
./eidosian_venv/bin/python -m pytest -q word_forge/tests
./eidosian_venv/bin/python -m pytest -q scripts/tests/test_living_knowledge_pipeline.py
```

Code Forge benchmark (scoped baseline):

```bash
./eidosian_venv/bin/code-forge benchmark \
  --root /path/to/interop-repo \
  --output reports/code_forge_benchmark_interop.json \
  --baseline reports/code_forge_benchmark_interop_baseline.json \
  --write-baseline
```

## Operational Notes

- Idempotent by design: each run writes to a timestamped run directory.
- Drift detection is deterministic and uses source-path + hash comparison.
- Binary files and oversized text files are skipped explicitly.
- All outputs are additive; no destructive rewrite of source repository files.
- For very large repositories with huge tracked non-source assets, run the pipeline on a scoped repo root (or curated mirror) to avoid multi-hour staging cycles.
- GraphRAG tooling uses `EIDOS_GRAPHRAG_ROOT` for workspace override and `EIDOS_GRAPHRAG_TIMEOUT_SEC` to prevent long-hanging query/index subprocess calls.

## External References

- GraphRAG docs and examples: https://microsoft.github.io/graphrag/
- GraphRAG repository: https://github.com/microsoft/graphrag
- Qodo public docs: https://docs.qodo.ai/
- Qodo public release/news context: https://www.qodo.ai/blog/
