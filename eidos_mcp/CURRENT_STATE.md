# Current State: eidos_mcp

**Date**: 2026-01-22
**Status**: Critical / Production

## 📊 Metrics
- **Role**: Middleware / Aggregator.
- **Complexity**: High (Depends on everything).

## 🏗️ Architecture
- `src/eidos_mcp/eidos_mcp_server.py`: Server bootstrap; registers resources and loads routers.
- `src/eidos_mcp/routers/`: Modular tool routers (system, memory, knowledge, gis, audit, diagnostics, types, nexus, **auth**).
- Uses `fastmcp` for decorator-based tool/resource definition.
- **Transports**: Supports Stdio, SSE, and StreamableHTTP.

## 🧠 Learnings & Observations
- StreamableHTTP requires running with `EIDOS_MCP_TRANSPORT=streamable-http` and a valid `FASTMCP_STREAMABLE_HTTP_PATH` (default `/streamable-http`).
- Stdio transport must avoid stdout noise; non-stdio logging now stays off stdout to keep JSON-RPC clean.
- Default ports are now in the 8928+ range to avoid common collisions (8000/8080/8928).

## 🛠️ Configuration
- Gemini CLI configured in `~/.gemini/settings.json`:
    - `eidosian_nexus`: Stdio (Default).
- `eidosian_nexus_sse`: SSE (`http://127.0.0.1:8928/sse`).
    - `eidosian_nexus_google`: SSE with `google_credentials` (Ready for ADC).
- Systemd service: `eidos-mcp.service` (User level, Port 8928).
- Auth Tokens: `~/.gemini/mcp-oauth-tokens.json` (Initialized).
