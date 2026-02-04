# Roadmap: computer_control_forge

## Current Version: 0.1.0 (Alpha)

---

## Phase 1: Foundation (v0.1.0 → v0.5.0) - Current

### Milestone 1.1: Core Stabilization ✅
- [x] Basic control service (keyboard, mouse, screen)
- [x] Kill switch and safety mechanisms
- [x] Wayland support via ydotool
- [x] Basic action logging

### Milestone 1.2: Perception Integration ✅
- [x] Screen capture (X11 + Wayland portal)
- [x] OCR engine integration (Tesseract)
- [x] Window manager integration (KWin DBus)
- [x] Unified perception state

### Milestone 1.3: Code Cleanup (In Progress)
- [ ] Consolidate mouse implementations
- [ ] Refactor to use lib/cli framework
- [ ] Add type hints throughout
- [ ] Document all public APIs

---

## Phase 2: Production Ready (v0.5.0 → v1.0.0)

### Milestone 2.1: Testing & Reliability
- [ ] Unit tests for all core modules
- [ ] Integration tests for perception pipeline
- [ ] Safety mechanism stress tests
- [ ] Performance benchmarks

### Milestone 2.2: MCP Integration
- [ ] Expose control tools via eidos_mcp
- [ ] Add permission model for remote control
- [ ] Implement action approval workflow
- [ ] Add tool usage analytics

### Milestone 2.3: Enhanced Perception
- [ ] Frame delta encoding (reduce bandwidth)
- [ ] Element tracking with persistent IDs
- [ ] UI element classification
- [ ] Application detection

---

## Phase 3: Advanced Features (v1.0.0 → v2.0.0)

### Milestone 3.1: Intelligent Actions
- [ ] Goal-directed action planning
- [ ] Error recovery and retry logic
- [ ] Complex gesture support
- [ ] Action macro system

### Milestone 3.2: Self-Verification
- [ ] Action verification via perception
- [ ] Automatic retry on failure
- [ ] Confidence scoring
- [ ] Failure pattern learning

### Milestone 3.3: Cross-Platform
- [ ] Abstract platform interface
- [ ] macOS backend
- [ ] Windows backend (WSL2)
- [ ] Headless mode for CI

---

## Phase 4: Embodied Intelligence (v2.0.0 → v3.0.0)

### Milestone 4.1: Learning System
- [ ] Learn from human demonstrations
- [ ] Adapt timing patterns
- [ ] Build application mental models
- [ ] Recognize error states

### Milestone 4.2: Application Adapters
- [ ] Browser adapter (DOM understanding)
- [ ] Terminal adapter (output parsing)
- [ ] IDE adapter (code navigation)
- [ ] Document adapter (PDF/Office)

### Milestone 4.3: Advanced Safety
- [ ] Hardware kill switch support
- [ ] Anomaly detection (unusual patterns)
- [ ] Sandboxed execution
- [ ] Replay and audit system

---

## Timeline

| Phase | Target Date | Status |
|-------|-------------|--------|
| 1.1 | 2026-01-23 | ✅ Complete |
| 1.2 | 2026-01-24 | ✅ Complete |
| 1.3 | 2026-02-01 | 🔶 In Progress |
| 2.1 | 2026-02-15 | ⬜ Planned |
| 2.2 | 2026-03-01 | ⬜ Planned |
| 2.3 | 2026-03-15 | ⬜ Planned |
| 3.x | 2026-Q2 | ⬜ Future |
| 4.x | 2026-Q3 | ⬜ Future |

---

## Dependencies

### External
- `pynput` - Keyboard/mouse control
- `mss` - Screen capture (X11)
- `tesseract` - OCR
- `ydotool` - Wayland control
- `dbus-python` - KWin integration

### Internal
- `eidos_mcp` - Tool exposure
- `llm_forge` - LLM integration
- `glyph_forge` - ASCII visualization
- `memory_forge` - State persistence

---

*Iterating towards embodied intelligence.*
