# ⚛️ EIDOSIAN NEXUS (MCP SERVER)

> _"The central nervous system of the Eidosian Forge."_

This directory contains the **Model Context Protocol (MCP)** server implementation for the Eidosian environment.

## 📡 SERVER DETAILS

*   **Entry Point**: `eidos_mcp_server.py`
*   **Transport**: Stdio (Standard Input/Output) by default.
*   **Dependencies**: `mcp`, `fastapi`, `uvicorn`.

## 💎 CAPABILITIES

### Resources (Data)
*   `eidos://persona` -> Returns the full text of `GEMINI.md`.
*   `eidos://roadmap` -> Returns `eidosian_roadmap.md`.
*   `eidos://todo` -> Returns `TODO.md`.

### Tools (Actions)
*   `codex_task(query)` -> Queues a task for the `codex` agent.
*   `remember(fact)` -> Saves a fact to `~/eidos_memory.json`.
*   `read_memory()` -> Retrieves stored memories.

### Prompts
*   `eidos_persona` -> Injects the Eidosian identity into the context.

## 🚀 USAGE

### Running Locally (Testing)
```bash
./run_server.sh
```

### Inspecting
```bash
mcp dev eidos_mcp_server.py
```

### Integration
Configure your MCP client (Claude Desktop, etc.) to run this script:

```json
{
  "mcpServers": {
    "eidosian-nexus": {
      "command": "/home/lloyd/eidosian_venv/bin/python3",
      "args": ["/home/lloyd/eidosian_forge/eidos_mcp/eidos_mcp_server.py"]
    }
  }
}
```

## 🔄 RECURSIVE GOAL
This server acts as the bridge between static files and dynamic agent intelligence. It allows any connected agent to "remember" and "act" within the Forge.
