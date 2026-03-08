# 💎 Audit Forge ⚡

> _"The Eyes of Eidos. Verifying integrity, tracking coverage, and ensuring standards."_

## 👁️ Overview

`audit_forge` provides the core infrastructure for systemic self-auditing within the Eidosian ecosystem. It tracks which files have been reviewed, orchestrates idempotent updates to task lists, and ensures compliance with Eidosian coding standards.

```ascii
      ╭───────────────────────────────────────────╮
      │               AUDIT FORGE                 │
      │    < Coverage | Tasks | Verification >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   COVERAGE MAP      │   │   TODO/ROADMAP  │
      │ (JSON State)        │   │ (Markdown)      │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Verification & Observability
- **Test Coverage**: 100% Core Logic Validated
- **Architecture**:
  - `audit_core.py`: Main orchestration (`AuditForge`).
  - `coverage.py`: Tracks file review status and timestamps (`CoverageTracker`).
  - `tasks.py`: Safely manages markdown-based task lists (`IdempotentTaskManager`).
  - `cli.py`: Typer-based CLI interface.

## 🚀 Usage & Workflows

### CLI Interface

Check coverage for the current directory:
```bash
python -m audit_forge.cli coverage
```

Mark a specific file as reviewed:
```bash
python -m audit_forge.cli mark "src/my_file.py" --agent eidos
```

Add an idempotent task to the global TODO:
```bash
python -m audit_forge.cli todo "Verify 100% type safety" --section "Immediate"
```

### Python API

```python
from pathlib import Path
from audit_forge.audit_core import AuditForge

audit = AuditForge(Path("./audit_data"))

# Verify what needs attention
stats = audit.verify_coverage("./src")
print(f"Unreviewed Files: {stats['unreviewed_count']}")

# Mark progress
audit.coverage.mark_reviewed("src/main.py", agent_id="eidos", scope="deep")
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Enhance CLI with rich output.
- [ ] Implement automated drift detection (hash-based invalidation of "reviewed" status).

### Future Vector
- Integrate with `knowledge_forge` to map audited files to semantic nodes.
- Expose an MCP tool explicitly for automated self-auditing sweeps.

---
*Generated and maintained by Eidos.*
