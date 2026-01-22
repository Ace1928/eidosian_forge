# Forge Dependency Map

Last update: 2026-01-22

## ASCII Topology

```ascii
EIDOS_MCP
├─ audit_forge      (audit_add_todo, audit_mark_reviewed)
├─ diagnostics_forge (diagnostics_ping, diagnostics_metrics)
├─ file_forge       (file_read, file_write, file ops)
├─ gis_forge        (gis_get, gis_set, config resource)
├─ knowledge_forge  (kb_add, grag_query, kb tools)
│  └─ graphrag      (GraphRAGIntegration)
├─ memory_forge     (memory_add, memory_retrieve)
├─ llm_forge        (agent reasoning, llm integration)
├─ agent_forge      (agent_run_task)
├─ refactor_forge   (refactor_analyze)
└─ type_forge       (type_register, type_validate)
```

## Structured Map (JSON)

```json
{
  "eidos_mcp": {
    "depends_on": [
      "audit_forge",
      "diagnostics_forge",
      "file_forge",
      "gis_forge",
      "knowledge_forge",
      "memory_forge",
      "llm_forge",
      "agent_forge",
      "refactor_forge",
      "type_forge",
      "graphrag"
    ],
    "tools": {
      "system": ["file_read", "file_write", "run_shell_command"],
      "gis": ["gis_get", "gis_set"],
      "knowledge": ["kb_add", "grag_query"],
      "memory": ["memory_add", "memory_retrieve"],
      "nexus": ["agent_run_task", "mcp_list_tools", "mcp_self_upgrade"],
      "audit": ["audit_add_todo", "audit_mark_reviewed"],
      "diagnostics": ["diagnostics_ping", "diagnostics_metrics"],
      "type": ["type_register", "type_validate"]
    },
    "resources": [
      "eidos://config",
      "eidos://persona",
      "eidos://roadmap",
      "eidos://todo"
    ]
  },
  "knowledge_forge": {
    "depends_on": ["graphrag"],
    "notes": "GraphRAGIntegration wraps graphrag for local queries."
  },
  "agent_forge": {
    "depends_on": ["llm_forge", "eidos_mcp.transactions"],
    "notes": "Uses LLM for planning; transactional file writes via MCP."
  }
}
```

## Structure Notes
- `eidos_mcp` uses a single `src/eidos_mcp` package layout (no root bridge).
- Dependent forges are expected to expose `src/<forge>` without root bridge shims.
