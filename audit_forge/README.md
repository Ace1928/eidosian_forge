# 💎 Audit Forge ⚡

```ascii
    ___    __  ______  __________     __________  ____  ____________
   /   |  / / / / __ \/  _/_  __/    / ____/ __ \/ __ \/ ____/ ____/
  / /| | / / / / / / // /  / /      / /_  / / / / /_/ / / __/ __/   
 / ___ |/ /_/ / /_/ // /  / /      / __/ / /_/ / _, _/ /_/ / /___   
/_/  |_|\____/_____/___/ /_/      /_/    \____/_/ |_|\____/_____/   
                                                                    
```

> _"The Eyes of Eidos. Verifying integrity, tracking coverage, and ensuring systemic compliance."_

## 👁️ Overview

`audit_forge` provides the core infrastructure for systemic self-auditing within the Eidosian ecosystem. It tracks review coverage across the entire filesystem, orchestrates idempotent updates to task lists, and ensures that every code artifact adheres to Eidosian standards. It serves as the primary mechanism for maintaining the **Elevation Roadmap** and preventing regression in the 🟢 Elevated status of all forges.

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Verification & Observability
- **Test Coverage**: 100% Core Logic verified.
- **Inference Standard**: **Qwen 3.5 2B** (Used for semantic task classification).
- **Core Architecture**:
  - **`audit_core.py`**: Main orchestration engine (`AuditForge`) integrating coverage and task management.
  - **`coverage.py`**: High-performance JSON-backed tracker for file-level review state and scope (shallow/deep).
  - **`tasks.py`**: Safely manages markdown-based `TODO.md` and `ROADMAP.md` files with non-destructive, idempotent updates.
  - **`cli.py`**: Rich Typer-based interface for interactive audit sessions.

## 🚀 Usage & Workflows

### Coverage Tracking

Check which files in the current scope require Eidosian review:
```bash
python -m audit_forge.cli coverage --root ./src
```

Verify and mark a specific module as 🟢 Elevated:
```bash
python -m audit_forge.cli mark "src/core/logic.py" --agent eidos --scope deep
```

### Idempotent Task Management

Add a task to the global roadmap without creating duplicates:
```bash
python -m audit_forge.cli todo "Verify 100% type safety in memory_forge" --section "Elevation"
```

## 🔗 System Integration

- **All Forges**: Used uniformly to track the elevation status reported in `FORGE_STATUS.md`.
- **Eidos MCP**: Exposes `audit_add_todo` and `audit_mark_reviewed` as standard tools for agents.
- **Repo Forge**: Informs the "doctor" command with coverage metrics to determine forge health.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy documentation into unified standard.
- [x] Integrate Eidosian observability decorators in the audit core.
- [ ] Implement automated "Hash-Based Invalidation" where a file's review status is revoked if its content hash changes.

### Future Vector (Phase 3+)
- Integrate with `knowledge_forge` to map audited files to semantic concept nodes, enabling "Evidence-Based RAG".
- Build an "Autonomous Auditor" daemon that scans for structural drift and automatically creates `TODO.md` entries for remediation.
- Implement multi-agent consensus for "Deep Review" marking.

---
*Generated and maintained by Eidos.*
