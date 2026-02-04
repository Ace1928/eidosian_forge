# Issues: web_interface_forge

**Last Updated**: 2026-01-25

---

## 🔴 Critical

None currently.

---

## 🟠 High Priority

### ISSUE-001: ChatGPT DOM Selector Brittleness
**Description**: ChatGPT UI changes frequently, breaking selectors.
**Impact**: Server fails to read messages or send text.
**Current Mitigation**: Multiple selector candidates in `ChatDOM.NEW_CHAT_CANDIDATES`.
**Proposed Solution**: 
- Implement ML-based element detection
- Multiple fallback strategies
- Automatic selector healing
**Status**: 🔶 Needs work

---

### ISSUE-002: No Test Suite
**Description**: Only stub test file exists, no actual tests.
**Impact**: Cannot verify functionality, no CI/CD.
**Status**: 🔶 Needs work

---

### ISSUE-003: No README
**Description**: Missing README.md with full documentation.
**Impact**: New users can't understand or use the forge.
**Status**: 🔶 Needs work

---

## 🟡 Medium Priority

### ISSUE-004: No StandardCLI Integration
**Description**: Doesn't use `lib/cli` StandardCLI framework.
**Impact**: Inconsistent CLI interface, no bash completions.
**Status**: ⬜ Planned

---

### ISSUE-005: Single Provider Only
**Description**: Only supports ChatGPT, no other AI providers.
**Impact**: Limited utility.
**Status**: ⬜ Planned

---

### ISSUE-006: No Configuration File
**Description**: All settings hardcoded or via CLI args.
**Impact**: Difficult to customize.
**Proposed Solution**: YAML/JSON config file support.
**Status**: ⬜ Planned

---

### ISSUE-007: CDP Connection Required
**Description**: Requires Chromium running with remote debugging on port 9222.
**Impact**: Manual setup step, not documented.
**Solution**: Auto-launch browser option.
**Status**: ⬜ Planned

---

## 🟢 Low Priority / Nice-to-Have

### ISSUE-008: No Headless Mode
**Description**: Requires headed browser (display).
**Proposed**: Support headless with virtual display.
**Status**: ⬜ Future

---

### ISSUE-009: No Docker Support
**Description**: No containerization.
**Impact**: Deployment complexity.
**Status**: ⬜ Future

---

### ISSUE-010: Limited Error Recovery
**Description**: Some error paths don't recover gracefully.
**Status**: ⬜ Future

---

## 🐛 Bugs

### BUG-001: Stream reset noise
**Severity**: Low
**Description**: "[stream reset]" messages appear in terminal during DOM reflows.
**Impact**: Minor visual noise.
**Status**: ⬜ Low priority

---

### BUG-002: Transcript duplication
**Severity**: Low  
**Description**: Edge case where messages can be recorded twice.
**Root Cause**: Hash collision or timing issue.
**Status**: ⬜ Needs investigation

---

## 💡 Enhancement Requests

### ENH-001: Conversation History
**Description**: Browse and search past conversations.
**Priority**: Medium
**Status**: ⬜ Planned

---

### ENH-002: Export Formats
**Description**: Export transcripts as JSON, HTML, PDF.
**Priority**: Low
**Status**: ⬜ Future

---

### ENH-003: Voice Input/Output
**Description**: Speech-to-text input, TTS output.
**Priority**: Low
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
