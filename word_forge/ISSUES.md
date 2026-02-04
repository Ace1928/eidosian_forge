# Issues: word_forge

**Last Updated**: 2026-01-25

---

## 🔴 Critical

None currently.

---

## 🟠 High Priority

### ISSUE-001: Massive config.py File
**Description**: `config.py` is 44,577 lines - should be split into modules.
**Impact**: Hard to maintain, slow to load, difficult to navigate.
**Proposed Solution**: 
- Split into `configs/` submodule
- Separate per-component configs
- Lazy loading where possible
**Status**: 🔶 Needs work

---

### ISSUE-002: NLTK Data Download on First Run
**Description**: First run triggers NLTK corpus downloads.
**Impact**: Poor first-run experience, network dependency.
**Proposed Solution**: 
- Pre-package common corpora
- Graceful degradation without corpora
- Better progress indication
**Status**: ⬜ Planned

---

### ISSUE-003: Memory Usage with Large Models
**Description**: Large embedding models consume significant RAM.
**Impact**: OOM on constrained systems.
**Current Mitigation**: Environment variable to select smaller models.
**Status**: ⬜ Planned

---

## 🟡 Medium Priority

### ISSUE-004: No MCP Tool Exposure
**Description**: Word lookup and graph query not exposed via MCP.
**Impact**: Can't use word_forge from MCP clients.
**Status**: ⬜ Planned

---

### ISSUE-005: Inconsistent Error Handling
**Description**: Some modules don't use `exceptions.py` consistently.
**Impact**: Unpredictable error behavior.
**Status**: ⬜ Planned

---

### ISSUE-006: Centralised Config Not Used
**Description**: Uses own config, not eidosian_forge centralised config.
**Impact**: Duplicate configuration, potential conflicts.
**Status**: ⬜ Planned

---

### ISSUE-007: Test Coverage Gaps
**Description**: Some modules have lower test coverage than others.
**Impact**: Potential bugs in less-tested areas.
**Status**: ⬜ Planned

---

## 🟢 Low Priority / Nice-to-Have

### ISSUE-008: No Streaming Support
**Description**: Graph operations are blocking, no streaming output.
**Proposed**: Async generators for large operations.
**Status**: ⬜ Future

---

### ISSUE-009: Limited Visualization Options
**Description**: Only PyVis and Plotly supported.
**Proposed**: Add D3.js, Cytoscape export.
**Status**: ⬜ Future

---

### ISSUE-010: No Web Interface
**Description**: CLI only, no web dashboard.
**Proposed**: Flask/FastAPI dashboard for exploration.
**Status**: ⬜ Future

---

## 🐛 Bugs

### BUG-001: SQLite Lock with Multiple Workers
**Severity**: Medium
**Description**: Occasional "database is locked" with parallel workers.
**Root Cause**: WAL mode helps but doesn't eliminate.
**Workaround**: Single-writer pattern.
**Status**: ⬜ Needs investigation

---

### BUG-002: Graph Visualization Memory Leak
**Severity**: Low
**Description**: Large graphs can cause memory growth in visualization.
**Impact**: Process memory grows over time.
**Status**: ⬜ Low priority

---

## 💡 Enhancement Requests

### ENH-001: Multi-Language Support
**Description**: Support for languages beyond English.
**Priority**: Medium
**Status**: ⬜ Future

---

### ENH-002: Real-Time Graph Updates
**Description**: Live graph updates as new words are added.
**Priority**: Low
**Status**: ⬜ Future

---

### ENH-003: Export to Standard Formats
**Description**: Export to RDF, GraphML, etc.
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

*Track issues. Fix issues. Forge words.*
