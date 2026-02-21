# Forge Interop Validation (2026-02-21)

## Scope

Validated cross-forge interoperability and production readiness for:

- `knowledge_forge`
- `memory_forge`
- `code_forge`
- `word_forge`
- GraphRAG integration path (`knowledge_forge` + `eidos_mcp` + `graphrag_workspace`)

## Implementation Updates

1. GraphRAG command compatibility hardening
- File: `knowledge_forge/src/knowledge_forge/integrations/graphrag.py`
- Added dual command support:
  - primary: `python -m graphrag <subcommand> ...`
  - fallback: `python -m graphrag.<subcommand> ...`
- Added explicit command/returncode metadata in responses.
- Added subprocess timeout guard (`EIDOS_GRAPHRAG_TIMEOUT_SEC`, default `900`) to prevent hung MCP calls.

2. MCP GraphRAG root correctness
- File: `eidos_mcp/src/eidos_mcp/routers/knowledge.py`
- Default root now resolves to:
  1. `EIDOS_GRAPHRAG_ROOT` (if set)
  2. `graphrag_workspace`
  3. legacy `graphrag`

3. GraphRAG workspace config migration
- File: `graphrag_workspace/settings.yaml`
- Migrated to current GraphRAG config schema (`completion_models`, `embedding_models`, `input_storage`, etc.).
- Fixed template-safe regex (`file_pattern: ".*\\.txt$$"`).
- Added tracked log directory sentinel: `graphrag_workspace/logs/.gitkeep`.

4. Documentation refresh
- File: `knowledge_forge/README.md` replaced with current architecture and integration details.
- File: `graphrag_workspace/README.md` updated command syntax (`python -m graphrag query/index ...`) and root override details.
- File: `eidos_mcp/README.md` added `EIDOS_GRAPHRAG_ROOT` configuration.
- File: `docs/LIVING_KNOWLEDGE_SYSTEM.md` expanded validation + benchmark guidance.

## Test and Validation Results

### Forge test suites

```bash
./eidosian_venv/bin/python -m pytest -q code_forge/tests memory_forge/tests word_forge/tests
```
- Result: `640 passed, 18 skipped`

```bash
./eidosian_venv/bin/python -m pytest -q knowledge_forge/tests \
  eidos_mcp/tests/test_routers.py \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py \
  scripts/tests/test_living_knowledge_pipeline.py
```
- Result: `42 passed, 1 skipped`

### Living knowledge pipeline

Executed on a scoped sample repository containing code/memory/knowledge/doc artifacts from all target forges:

```bash
./eidosian_venv/bin/python scripts/living_knowledge_pipeline.py \
  --repo-root <scoped_repo> \
  --output-root /data/data/com.termux/files/usr/tmp/living_knowledge_reports \
  --workspace-root /data/data/com.termux/files/usr/tmp/living_knowledge_workspace \
  --code-max-files 50 \
  --max-file-bytes 200000 \
  --max-chars-per-doc 4000
```
- Result: `ok: true`, `records_total: 10`

### Code Forge benchmark (scoped baseline)

```bash
./eidosian_venv/bin/code-forge benchmark \
  --root <scoped_repo> \
  --output reports/code_forge_benchmark_interop_20260221.json \
  --baseline reports/code_forge_benchmark_interop_baseline.json \
  --write-baseline
```
- Result: regression gate passed (`pass: true`)

### Code Forge benchmark (knowledge_forge source)

```bash
./eidosian_venv/bin/code-forge benchmark \
  --root knowledge_forge/src \
  --max-files 20 \
  --ingestion-repeats 1 \
  --query-repeats 1 \
  --output reports/code_forge_benchmark_knowledge_forge_20260221.json \
  --baseline reports/code_forge_benchmark_knowledge_forge_baseline.json \
  --write-baseline
```
- Result: regression gate passed (`pass: true`)
- Ingestion: 6 files, 212 units, 17.53s (`0.34 files/s`)
- Search mean latency: 58.31ms
- Dependency graph: 422 nodes / 1055 edges

### Living knowledge pipeline (scoped interop run)

```bash
./eidosian_venv/bin/python scripts/living_knowledge_pipeline.py \
  --repo-root knowledge_forge \
  --output-root reports/living_knowledge_interop_20260221 \
  --workspace-root graphrag_workspace \
  --code-max-files 40 \
  --max-file-bytes 250000 \
  --max-chars-per-doc 4000
```
- Result: `ok: true`, `records_total: 65`

## Operational Notes

- GraphRAG index/query now execute through the correct modern CLI entrypoint.
- Current GraphRAG failure mode (if present) is runtime backend connectivity (`api_base` endpoint unavailable), not schema/command wiring.
- If MCP server is already running, restart it to load the updated router and GraphRAG root defaults.
