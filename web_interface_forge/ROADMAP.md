# Roadmap: web_interface_forge

## Current Version: 0.1.0 (Alpha)

---

## Phase 1: Foundation (v0.1.0 → v0.5.0) - Current

### Milestone 1.1: Core Functionality ✅
- [x] Playwright browser automation
- [x] WebSocket server/client
- [x] DOM scanning and message extraction
- [x] Delta streaming for assistant output
- [x] Storage state persistence

### Milestone 1.2: Stabilization (In Progress)
- [ ] Add comprehensive test suite
- [ ] Implement StandardCLI framework
- [ ] Add configuration file support
- [ ] Create full README documentation
- [ ] Handle selector changes gracefully

---

## Phase 2: Production Ready (v0.5.0 → v1.0.0)

### Milestone 2.1: Robustness
- [ ] Multiple selector strategies per element
- [ ] Automatic reconnection with exponential backoff
- [ ] Error recovery and retry logic
- [ ] Health check endpoint

### Milestone 2.2: MCP Integration
- [ ] Expose chat as MCP tool
- [ ] Async message queuing
- [ ] Response streaming to MCP clients
- [ ] Rate limiting

### Milestone 2.3: Testing
- [ ] Unit tests for all modules
- [ ] Integration tests with mocked browser
- [ ] End-to-end tests (manual/CI)
- [ ] Performance benchmarks

---

## Phase 3: Multi-Provider (v1.0.0 → v2.0.0)

### Milestone 3.1: Provider Abstraction
- [ ] Define provider interface
- [ ] Factor ChatGPT-specific code
- [ ] Create provider registry

### Milestone 3.2: Additional Providers
- [ ] Claude web UI adapter
- [ ] Gemini web UI adapter
- [ ] Perplexity adapter
- [ ] Generic web form adapter

### Milestone 3.3: Session Management
- [ ] Multiple concurrent sessions
- [ ] Session switching
- [ ] Session naming/tagging
- [ ] Cross-session search

---

## Phase 4: Autonomous Agent (v2.0.0 → v3.0.0)

### Milestone 4.1: Navigation
- [ ] URL navigation commands
- [ ] Form filling
- [ ] Click/scroll actions
- [ ] Screenshot capture

### Milestone 4.2: Data Extraction
- [ ] Structured data extraction
- [ ] Table parsing
- [ ] PDF handling
- [ ] Content summarization

### Milestone 4.3: Workflows
- [ ] Multi-step task definitions
- [ ] Conditional logic
- [ ] Error handling and recovery
- [ ] Workflow templates

---

## Timeline

| Phase | Target Date | Status |
|-------|-------------|--------|
| 1.1 | 2026-01-25 | ✅ Complete |
| 1.2 | 2026-02-01 | 🔶 In Progress |
| 2.1 | 2026-02-15 | ⬜ Planned |
| 2.2 | 2026-03-01 | ⬜ Planned |
| 2.3 | 2026-03-15 | ⬜ Planned |
| 3.x | 2026-Q2 | ⬜ Future |
| 4.x | 2026-Q3 | ⬜ Future |

---

## Dependencies

### External
- `playwright` - Browser automation
- `websockets` - WebSocket server/client
- `eidosian_core` - Decorators and logging

### Internal
- `eidos_mcp` - Tool exposure
- `memory_forge` - Conversation persistence
- `knowledge_forge` - Transcript indexing

---

*Bridging web interfaces to machine intelligence.*
