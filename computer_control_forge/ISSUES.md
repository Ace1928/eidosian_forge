# Issues: computer_control_forge

**Last Updated**: 2026-01-25

---

## 🔴 Critical

None currently.

---

## 🟠 High Priority

### ISSUE-001: Mouse Implementation Consolidation
**Description**: 7 different mouse implementations exist, creating confusion and maintenance burden.
**Files Affected**:
- `absolute_mouse.py`
- `advanced_mouse.py`
- `calibrated_mouse.py`
- `human_mouse.py`
- `live_mouse.py`
- `perception_mouse.py`
- `pid_mouse.py`

**Proposed Solution**: Consolidate into 2 implementations:
1. `standard_mouse.py` - Direct positioning
2. `natural_mouse.py` - Human-like movement with PID control

**Status**: 🔶 Needs work

---

### ISSUE-002: Wayland Screenshot Latency
**Description**: Portal-based screenshot capture has ~1s delay due to file system detection.
**Root Cause**: Using file-based screenshot detection instead of direct capture.
**Proposed Solution**: Use `grim` for direct Wayland capture, fall back to portal.
**Status**: 🔶 Needs work

---

### ISSUE-003: Low Test Coverage
**Description**: Only `test_safety.py` exists. Core functionality untested.
**Affected Areas**:
- `control.py` - 0 tests
- `perception/*.py` - 0 tests
- Mouse implementations - 0 tests

**Status**: 🔶 Needs work

---

## 🟡 Medium Priority

### ISSUE-004: No CLI Framework Integration
**Description**: Doesn't use `lib/cli` StandardCLI framework like other forges.
**Impact**: Inconsistent CLI interface, no bash completions.
**Status**: ⬜ Planned

---

### ISSUE-005: Missing Type Hints
**Description**: Several modules lack comprehensive type hints.
**Files**: `agent_loop.py`, various mouse implementations
**Status**: ⬜ Planned

---

### ISSUE-006: OCR Performance
**Description**: Tesseract is slow on large screen regions.
**Impact**: Delays in perception pipeline.
**Proposed Solution**: 
- Region-based OCR (only changed areas)
- Parallel OCR processing
- Consider EasyOCR as alternative
**Status**: ⬜ Planned

---

### ISSUE-007: No MCP Tool Exposure
**Description**: Control tools not exposed via eidos_mcp.
**Impact**: Cannot control computer via MCP interface.
**Status**: ⬜ Planned

---

## 🟢 Low Priority / Nice-to-Have

### ISSUE-008: Visual Feedback Overlay
**Description**: No visual indication of where actions are happening.
**Proposed**: Draw cursor position, click locations, typed text on screen overlay.
**Status**: ⬜ Future

---

### ISSUE-009: Action Replay System
**Description**: Cannot replay logged actions for debugging.
**Proposed**: Load session JSON and replay actions with timing.
**Status**: ⬜ Future

---

### ISSUE-010: Cross-Platform Support
**Description**: Only works on Linux (X11/Wayland).
**Desired**: macOS, Windows support.
**Status**: ⬜ Future

---

## 🐛 Bugs

### BUG-001: KillSwitchEngaged not always propagated
**Severity**: Medium
**Description**: In some edge cases, kill switch state isn't checked before action completion.
**Reproduction**: Engage kill switch during long `type_text` operation.
**Status**: ⬜ Needs investigation

---

### BUG-002: PID file not cleaned on crash
**Severity**: Low
**Description**: If process crashes, PID file remains.
**Impact**: Stale PID file, need manual cleanup.
**Workaround**: `rm /tmp/eidosian_control.pid`
**Status**: ⬜ Needs fix

---

## 💡 Enhancement Requests

### ENH-001: Gesture Support
**Description**: Add drag, scroll, double-click, right-click gestures.
**Priority**: Medium
**Status**: ⬜ Planned

---

### ENH-002: Action Macros
**Description**: Define and execute sequences of actions.
**Priority**: Low
**Status**: ⬜ Future

---

### ENH-003: Application-Aware Control
**Description**: Detect current application and adapt behavior.
**Priority**: Medium
**Status**: ⬜ Future

---

## 📊 Issue Summary

| Category | Count |
|----------|-------|
| Critical | 0 |
| High | 3 |
| Medium | 4 |
| Low | 3 |
| Bugs | 2 |
| Enhancements | 3 |
| **Total** | **15** |

---

*Track issues. Fix issues. Ship quality.*
