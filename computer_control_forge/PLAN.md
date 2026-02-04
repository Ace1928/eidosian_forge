# Plan: computer_control_forge

**Objective**: Stabilize and productionize the computer control forge.

---

## Current Sprint: Code Cleanup & Testing

### Phase A: Mouse Consolidation (Priority 1)

**Goal**: Reduce 7 mouse implementations to 2.

**Tasks**:
- [ ] Analyze all mouse implementations for unique features
- [ ] Design unified `MouseController` interface
- [ ] Implement `StandardMouse` (direct positioning)
- [ ] Implement `NaturalMouse` (human-like movement from `human_mouse.py` + PID from `pid_mouse.py`)
- [ ] Update `control.py` to use new mouse classes
- [ ] Deprecate old implementations
- [ ] Add migration guide

**Acceptance Criteria**:
- Two mouse classes only
- All existing functionality preserved
- Tests pass

---

### Phase B: Test Suite (Priority 2)

**Goal**: Achieve 70%+ test coverage.

**Tasks**:
- [ ] Add tests for `ControlService`:
  - [ ] `test_start_stop`
  - [ ] `test_type_text`
  - [ ] `test_click`
  - [ ] `test_capture_screen`
  - [ ] `test_rate_limiting`
  - [ ] `test_dry_run_mode`
  - [ ] `test_session_logging`
  
- [ ] Add tests for perception:
  - [ ] `test_screen_capture`
  - [ ] `test_change_detection`
  - [ ] `test_ocr_engine`
  - [ ] `test_window_manager`
  
- [ ] Add integration tests:
  - [ ] `test_perception_action_loop`
  - [ ] `test_wayland_control`

**Acceptance Criteria**:
- `pytest --cov` shows 70%+ coverage
- All tests run without actual screen interaction (mocked backends)
- Integration tests marked for manual execution

---

### Phase C: CLI Integration (Priority 3)

**Goal**: Use StandardCLI framework.

**Tasks**:
- [ ] Create `cli.py` extending `StandardCLI`
- [ ] Implement commands:
  - [ ] `status` - Show control service status
  - [ ] `capture` - Take screenshot
  - [ ] `click X Y` - Click at position
  - [ ] `type TEXT` - Type text
  - [ ] `kill` - Engage kill switch
  - [ ] `safe` - Disengage kill switch
- [ ] Generate bash completions
- [ ] Update `bin/eidosian` to include control forge
- [ ] Add `INSTALL.md` updates

**Acceptance Criteria**:
- `eidosian control status` works
- Tab completion works
- Help text is comprehensive

---

## Dependencies

| Task | Depends On |
|------|------------|
| Mouse consolidation | None |
| Test suite | Mouse consolidation (for clean testing) |
| CLI integration | Test suite (verify stability first) |

---

## Timeline

| Phase | Duration | Target Completion |
|-------|----------|-------------------|
| A: Mouse | 2 days | 2026-01-27 |
| B: Tests | 3 days | 2026-01-30 |
| C: CLI | 2 days | 2026-02-01 |

---

## Success Criteria

1. **Code Quality**
   - No duplicate mouse implementations
   - Type hints on all public APIs
   - Docstrings on all modules

2. **Testing**
   - 70%+ coverage
   - All tests pass
   - CI integration

3. **Usability**
   - CLI matches other forges
   - Comprehensive help
   - Bash completions

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Comprehensive tests before refactor |
| Wayland compatibility issues | Maintain fallback to portal |
| Time overrun | Prioritize mouse + tests, defer CLI |

---

*Execute with precision. Verify with tests.*
