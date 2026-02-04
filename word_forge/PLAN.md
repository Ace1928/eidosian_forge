# Plan: word_forge

**Objective**: Integrate word_forge with eidosian_forge ecosystem and refactor for maintainability.

---

## Current Sprint: Documentation & Integration

### Phase A: Documentation (Priority 1) ✅

**Goal**: Complete standard documentation set.

**Tasks**:
- [x] Create CURRENT_STATE.md
- [x] Create GOALS.md
- [x] Create ROADMAP.md
- [x] Create ISSUES.md
- [x] Create PLAN.md

**Acceptance Criteria**:
- All docs follow Eidosian template
- Accurate metrics and status

---

### Phase B: Config Refactoring (Priority 2)

**Goal**: Split 44k LOC config.py into manageable modules.

**Tasks**:
- [ ] Analyze config.py structure
- [ ] Identify logical groupings
- [ ] Create `configs/` submodule structure:
  - [ ] `configs/database_config.py`
  - [ ] `configs/graph_config.py`
  - [ ] `configs/emotion_config.py`
  - [ ] `configs/vector_config.py`
  - [ ] `configs/parser_config.py`
  - [ ] `configs/queue_config.py`
  - [ ] `configs/base_config.py`
- [ ] Update imports throughout codebase
- [ ] Verify all tests pass

**Acceptance Criteria**:
- No single config file >2000 LOC
- All tests pass
- No import changes for external users

---

### Phase C: MCP Integration (Priority 3)

**Goal**: Expose word_forge functionality via eidos_mcp.

**Tasks**:
- [ ] Create MCP router in `eidos_mcp/routers/word_forge.py`
- [ ] Implement tools:
  - [ ] `word_lookup` - Get word definition and relationships
  - [ ] `word_search` - Semantic search
  - [ ] `graph_query` - Query semantic graph
  - [ ] `emotion_analyze` - Analyze text emotion
- [ ] Add to plugin registry
- [ ] Test via MCP client

**Acceptance Criteria**:
- Tools accessible via MCP
- Documentation updated
- Integration tests pass

---

### Phase D: Centralised Config (Priority 4)

**Goal**: Use eidosian_forge centralised configuration.

**Tasks**:
- [ ] Identify shared config values
- [ ] Create adapter for central config
- [ ] Maintain backwards compatibility
- [ ] Document config precedence

**Acceptance Criteria**:
- Uses `data/model_config.json` where applicable
- Local overrides still work
- No breaking changes

---

## Dependencies

| Task | Depends On |
|------|------------|
| Documentation | None |
| Config refactoring | Documentation |
| MCP integration | Config refactoring |
| Centralised config | MCP integration |

---

## Timeline

| Phase | Duration | Target Completion |
|-------|----------|-------------------|
| A: Documentation | 1 day | ✅ 2026-01-25 |
| B: Config refactor | 3 days | 2026-01-28 |
| C: MCP integration | 2 days | 2026-01-30 |
| D: Central config | 1 day | 2026-01-31 |

---

## Success Criteria

1. **Documentation**
   - All 5 standard docs complete ✅

2. **Code Quality**
   - Config split into modules
   - No file >2000 LOC

3. **Integration**
   - MCP tools functional
   - Central config used

---

## Notes

- word_forge has its own git history (originally separate repo)
- Comprehensive test suite exists (34 files)
- CI/CD already configured
- Pre-commit hooks in place

---

*Document. Refactor. Integrate. Forge.*
