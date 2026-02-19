# Port Registry

This is the canonical local port contract for Eidosian Forge services.

- Machine-readable source of truth: `config/ports.json`
- Override mechanism: service-specific environment variables
- Fallback rule: if env var is unset/empty, service must read `config/ports.json`

## Core Range

- Reserved Eidosian service range: `8928-8998` (step `2`)
- Current allocations:
  - `8928`: `eidos_mcp` (HTTP MCP endpoint on `/mcp`)
  - `8930`: `doc_forge_dashboard` (Doc processor API + live dashboard)
  - `8932`: `web_interface_sidecar_ws`
  - `8934`: `web_interface_http`

## Service Map

| Service Key | Host | Port | Protocol | Default Path | Primary Env Override(s) |
| --- | --- | ---: | --- | --- | --- |
| `eidos_mcp` | `127.0.0.1` | `8928` | `http` | `/mcp` | `FASTMCP_PORT`, `EIDOS_MCP_PORT` |
| `doc_forge_dashboard` | `127.0.0.1` | `8930` | `http` | `/` | `EIDOS_DOC_FORGE_PORT` |
| `web_interface_sidecar_ws` | `127.0.0.1` | `8932` | `ws` | `/` | `EIDOS_WEB_INTERFACE_WS_PORT` |
| `web_interface_http` | `127.0.0.1` | `8934` | `http` | `/` | `EIDOS_WEB_INTERFACE_HTTP_PORT` |
| `graphrag_llm` | `127.0.0.1` | `8081` | `http` | `/completion` | `EIDOS_GRAPHRAG_LLM_PORT` |
| `graphrag_embedding` | `127.0.0.1` | `8082` | `http` | `/embedding` | `EIDOS_GRAPHRAG_EMBED_PORT` |
| `graphrag_judges_base` | `127.0.0.1` | `8091` | `http` | `/completion` | `EIDOS_GRAPHRAG_JUDGE_PORT_BASE` |
| `doc_forge_llm` | `127.0.0.1` | `8093` | `http` | `/completion` | `EIDOS_DOC_FORGE_LLM_PORT` |
| `file_forge_embedding_proxy` | `127.0.0.1` | `11435` | `http` | `/v1/embeddings` | `EIDOS_FILE_FORGE_EMBED_PROXY_PORT` |
| `ollama_http` | `127.0.0.1` | `11434` | `http` | `/` | `EIDOS_OLLAMA_PORT` |

## Validation

To inspect effective values:

```bash
./eidosian_venv/bin/python scripts/port_registry.py dump
./eidosian_venv/bin/python scripts/port_registry.py get --service eidos_mcp --field port
./eidosian_venv/bin/python scripts/port_registry.py url --service eidos_mcp
```

## Change Policy

1. Update `config/ports.json` first.
2. Update this document and any service docs that expose a default port.
3. Keep MCP and doc dashboard inside the Eidosian reserved range.
4. Avoid conflicts with ephemeral bench servers (`8081/8082/8091/8093`).
