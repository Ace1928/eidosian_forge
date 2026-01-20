# TODO: eidos_mcp

## 🚨 High Priority
- [ ] **Refactor**: Split `eidos_mcp_server.py` into multiple files (routers) per Forge to reduce the massive file size.
- [ ] **Dependency Management**: Ensure all local forges are importable (editable installs).

## 🟡 Medium Priority
- [ ] **Testing**: Mock the other Forges to test the MCP server in isolation.
- [ ] **Configuration**: Unify config loading completely via `gis_forge`.

## 🟢 Low Priority
- [ ] **ChatMock**: Evaluate if `ChatMock` should be its own Forge or stay in `projects/`.