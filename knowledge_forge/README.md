# Knowledge Forge

`knowledge_forge` provides the persistent knowledge graph layer used by MCP, Code Forge integrations, and memory/knowledge bridge workflows.

## Core Modules

- `knowledge_forge/src/knowledge_forge/core/graph.py`
  - `KnowledgeForge`
  - `KnowledgeNode`
  - graph CRUD (`add_knowledge`, `search`, `get_by_tag`, `get_by_concept`, `link_nodes`, `stats`)
- `knowledge_forge/src/knowledge_forge/core/bridge.py`
  - `KnowledgeMemoryBridge`
  - unified memory+knowledge search and memory→knowledge promotion
- `knowledge_forge/src/knowledge_forge/integrations/graphrag.py`
  - GraphRAG index/query integration
  - compatibility runner for both:
    - `python -m graphrag <subcommand> ...` (current)
    - `python -m graphrag.<subcommand> ...` (legacy fallback)
- `knowledge_forge/src/knowledge_forge/integrations/memory_ingest.py`
  - ingest memory artifacts into graph nodes

## Integration Points

- MCP router: `eidos_mcp/src/eidos_mcp/routers/knowledge.py`
  - `kb_*`, `memory_*_semantic`, `grag_*`, `unified_context_search`, `promote_memory_to_knowledge`
- Code Forge:
  - `code_forge` sync pipeline writes knowledge links and provenance registry records.
- Living knowledge pipeline:
  - `scripts/living_knowledge_pipeline.py` stages KB + memory + code/doc corpus for GraphRAG.

## GraphRAG Root Resolution

MCP knowledge router uses this precedence:

1. `EIDOS_GRAPHRAG_ROOT` (if set)
2. `<forge>/graphrag_workspace` (default)
3. `<forge>/graphrag` (legacy fallback)

GraphRAG subprocess timeout is controlled by:

- `EIDOS_GRAPHRAG_TIMEOUT_SEC` (default: `900`, minimum: `30`)

## CLI

```bash
./eidosian_venv/bin/knowledge-forge status
./eidosian_venv/bin/knowledge-forge search "workspace competition"
./eidosian_venv/bin/knowledge-forge unified "memory bridge"

# RDF import/export (requires `knowledge_forge[rdf]`)
./eidosian_venv/bin/knowledge-forge export-rdf ./data/kb.ttl --format turtle
./eidosian_venv/bin/knowledge-forge import-rdf ./data/kb.ttl --format turtle

# Interactive graph visualization (requires `knowledge_forge[viz]`)
./eidosian_venv/bin/knowledge-forge visualize ./reports/kb_graph.html --max-nodes 300
```

## Test

```bash
./eidosian_venv/bin/python -m pytest -q knowledge_forge/tests
```

Current suite status in this environment:
- passing tests with one intentional skip (`test_graphrag_integration`, requires fully provisioned GraphRAG runtime).
