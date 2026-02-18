# Plan: eidos_mcp

**Objective**: Production hardening and documentation.

---

## Current Sprint: Documentation & Testing

### Phase A: Documentation (Priority 1)

- [x] Create ROADMAP.md
- [x] Create ISSUES.md
- [x] Create PLAN.md
- [ ] Update README.md
- [ ] Document all routers
- [ ] Add API reference

### Phase B: Testing (Priority 2)

- [ ] Audit current test coverage
- [ ] Add router unit tests
- [ ] Add integration tests
- [ ] Mock external dependencies

### Phase C: Hardening (Priority 3)

- [ ] Error handling audit
- [ ] Logging standardization
- [ ] Add health endpoints
- [ ] Performance profiling

---

## Dependencies

- All integrated forges (gis, audit, memory, knowledge, etc.)
- FastMCP library
- External services (Ollama, Tika)

---

## Timeline

| Phase | Duration | Target |
|-------|----------|--------|
| A | 2 days | 2026-01-27 |
| B | 3 days | 2026-01-30 |
| C | 2 days | 2026-02-01 |

---

## Success Criteria

1. All routers documented
2. 90% test coverage
3. Health endpoints working
4. No stdout pollution

---

*Document. Test. Harden.*
