# TODO: figlet_forge

## 🟢 Low Priority
- [ ] **Consolidate**: Move `figlet_core.py` features into `src/figlet_forge`.
- [ ] **Fonts**: Add custom Eidosian FIGlet fonts.

## Test
- [ ] Transactional reliability test

## MCP Validation
- [ ] Sandbox audit test

## test item
- [ ] normal

## Self-Directed Review
- [ ] [EIDOS] Review and ingest concepts from archive_forge/eidos_v1_concept into Knowledge Graph.
- [ ] [EIDOS] Refactor legacy_scraper.py to use CrawlForge standards.

## Documentation
- [ ] Document 'PYTHONPATH=eidosian_forge/eidos_mcp/src' requirement for manual Stdio testing of eidos_mcp_server.py in DEBUGGING.md

## Bug Fixes
- [ ] Fix eidos_remember_self: TieredMemorySystem.remember_self() raises TypeError because it does not accept the 'importance' argument defined in the MCP schema.
- [ ] Fix path resolution in refactor_analyze: It incorrectly prepends the working directory to already relative paths.
