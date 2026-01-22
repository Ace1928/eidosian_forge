# Current State: eidos_mcp

**Date**: 2026-01-20
**Status**: Critical / Production

## 📊 Metrics
- **Role**: Middleware / Aggregator.
- **Complexity**: High (Depends on everything).

## 🏗️ Architecture
- `src/eidos_mcp/eidos_mcp_server.py`: Server bootstrap; registers resources and loads routers.
- `src/eidos_mcp/routers/`: Modular tool routers (system, memory, knowledge, gis, audit, diagnostics, types, nexus).
- Uses `fastmcp` for decorator-based tool/resource definition.

## 🐛 Known Issues
- Hardcoded paths to `HOME_DIR` and `FORGE_DIR` might be brittle if moved.
- Import block at the top is massive; arguably should be dynamic or plugin-based.
