# Plan: web_interface_forge

**Objective**: Stabilize and document the web interface forge.

---

## Current Sprint: Documentation & Testing

### Phase A: Documentation (Priority 1)

**Goal**: Complete all missing documentation.

**Tasks**:
- [x] Create CURRENT_STATE.md
- [x] Create GOALS.md
- [x] Create ROADMAP.md
- [x] Create ISSUES.md
- [x] Create PLAN.md
- [ ] Create comprehensive README.md
  - [ ] Overview and purpose
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Configuration options
  - [ ] Troubleshooting guide

**Acceptance Criteria**:
- All docs follow Eidosian template
- README enables new users to start quickly

---

### Phase B: Test Suite (Priority 2)

**Goal**: Achieve 60%+ test coverage.

**Tasks**:
- [ ] Test utilities:
  - [ ] `test_sha256_text`
  - [ ] `test_lcp_len`
  - [ ] `test_safe_json`
  - [ ] `test_ts_str`
  
- [ ] Test ChatDOM (with mocked page):
  - [ ] `test_wait_for_chat_ready`
  - [ ] `test_read_messages`
  - [ ] `test_send_text`
  
- [ ] Test ClientState:
  - [ ] `test_record_message`
  - [ ] `test_handle_messages`
  - [ ] `test_handle_assistant_stream`
  - [ ] `test_handle_assistant_stable`
  - [ ] `test_save_markdown`

**Acceptance Criteria**:
- `pytest --cov` shows 60%+ coverage
- Tests don't require actual browser

---

### Phase C: CLI Integration (Priority 3)

**Goal**: Use StandardCLI framework.

**Tasks**:
- [ ] Create `cli.py` extending `StandardCLI`
- [ ] Implement commands:
  - [ ] `status` - Show server status
  - [ ] `serve` - Start server
  - [ ] `connect` - Start client
  - [ ] `transcript` - View/export transcripts
- [ ] Generate bash completions
- [ ] Update `bin/eidosian` to include web forge

**Acceptance Criteria**:
- `eidosian web serve` works
- Tab completion works

---

### Phase D: Configuration (Priority 4)

**Goal**: Support config file.

**Tasks**:
- [ ] Define config schema (YAML)
- [ ] Create default config
- [ ] Load config in server/client
- [ ] Document config options

**Config Schema**:
```yaml
web_interface:
  server:
    host: 127.0.0.1
    port: 8928
    state_path: ~/.eidos_chatgpt_state.json
    log_path: ~/.eidos_sidecar_server.log
  
  client:
    ws_url: ws://127.0.0.1:8928
    reconnect_delay: 1.0
    max_reconnect_delay: 10.0
  
  chatgpt:
    url: https://chat.openai.com/
    scan_interval: 0.2
    stable_window: 0.9
```

**Acceptance Criteria**:
- Config file loaded if present
- CLI args override config

---

## Dependencies

| Task | Depends On |
|------|------------|
| Documentation | None |
| Test suite | Documentation |
| CLI integration | Test suite |
| Configuration | CLI integration |

---

## Timeline

| Phase | Duration | Target Completion |
|-------|----------|-------------------|
| A: Documentation | 1 day | 2026-01-25 |
| B: Tests | 2 days | 2026-01-27 |
| C: CLI | 1 day | 2026-01-28 |
| D: Config | 1 day | 2026-01-29 |

---

## Success Criteria

1. **Documentation**
   - All 5 standard docs complete
   - README enables quick start

2. **Testing**
   - 60%+ coverage
   - CI integration ready

3. **Usability**
   - Standard CLI commands
   - Config file support
   - Bash completions

---

*Document first. Test second. Ship quality.*
