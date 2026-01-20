# Current State: eidos_mcp

**Date**: 2026-01-20
**Status**: Critical / Production

## 📊 Metrics
- **Role**: Middleware / Aggregator.
- **Complexity**: High (Depends on everything).

## 🏗️ Architecture
- `eidos_mcp_server.py`: Monolithic server file defining all tools.
- Uses `fastmcp` for easy decorator-based tool definition.

## 🐛 Known Issues
- Hardcoded paths to `HOME_DIR` and `FORGE_DIR` might be brittle if moved.
- Import block at the top is massive; arguably should be dynamic or plugin-based.