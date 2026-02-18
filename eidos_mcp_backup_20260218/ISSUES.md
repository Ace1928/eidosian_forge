# Issues: eidos_mcp

**Last Updated**: 2026-01-25

---

## 🔴 Critical

### ISSUE-001: Stdio Stdout Pollution
**Description**: Any print() or logging to stdout breaks JSON-RPC.
**Impact**: Transport failures, dropped messages.
**Status**: ✅ Mitigated (redirected logging)

---

## 🟠 High Priority

### ISSUE-002: Router Discovery Manual
**Description**: All routers must be explicitly imported.
**Impact**: Adding new tools requires code changes.
**Status**: 🔶 Needs auto-discovery

### ISSUE-003: Test Coverage Low
**Description**: Many routers lack unit tests.
**Status**: 🔶 Needs work

### ISSUE-004: No Rate Limiting
**Description**: Unlimited requests per client.
**Impact**: Potential resource exhaustion.
**Status**: ⬜ Planned

---

## 🟡 Medium Priority

### ISSUE-005: Schema Not Auto-Generated
**Description**: OpenAPI/JSON-Schema not automatic.
**Status**: ⬜ Planned

### ISSUE-006: No Health Dashboard
**Description**: No visibility into router status.
**Status**: ⬜ Planned

---

## 🟢 Low Priority

### ISSUE-007: Plugin Hot-Reload
**Description**: Requires restart for new routers.
**Status**: ⬜ Future

---

## 📊 Summary

| Category | Count |
|----------|-------|
| Critical | 1 (mitigated) |
| High | 3 |
| Medium | 2 |
| Low | 1 |
| **Total** | **7** |

---

*Track issues. Fix issues. Connect systems.*
