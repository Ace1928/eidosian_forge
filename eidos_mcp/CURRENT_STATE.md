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
- **Transports**: Supports Stdio and SSE.

## 🐛 Known Issues
- Hardcoded paths to `HOME_DIR` and `FORGE_DIR` might be brittle if moved.
- Import block at the top is massive; arguably should be dynamic or plugin-based.

## 🛠️ Configuration
- Gemini CLI configured in `~/.gemini/settings.json`:
    - `eidosian_nexus`: Stdio (Default).
    - `eidosian_nexus_sse`: SSE (`http://127.0.0.1:8765/sse`).
    - `eidosian_nexus_google`: SSE with `google_credentials` (Ready for ADC).
- Systemd service: `eidos-mcp.service` (User level, Port 8765).
- Auth Tokens: `~/.gemini/mcp-oauth-tokens.json` (Initialized).