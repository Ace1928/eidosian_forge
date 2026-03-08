# ⏳ Version Forge ⚡

> _"The Timekeeper of Eidos. Stability requires a name; change requires a number."_

## 🧠 Overview

`version_forge` is the definitive authority on semantic versioning and compatibility enforcement within the Eidosian ecosystem. It ensures that cross-forge dependencies remain consistent, backwards compatibility is verified, and the evolutionary history of the system is strictly traceable.

```ascii
      ╭───────────────────────────────────────────╮
      │             VERSION FORGE                 │
      │    < SemVer | Compatibility | History >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   SEMVER ENGINE     │   │ DEPENDENCY MAP  │
      │ (Strict Parsing)    │   │ (Matrix Checks) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Dependency & Version Management
- **Test Coverage**: Core SemVer logic passing.
- **Architecture**:
  - `core.version.py`: Strict SemVer 2.0 implementation.
  - `compatibility/matrix.py`: Verifies component versions against system-wide constraint requirements.
  - `cli.py`: Typer-based command interface.

## 🚀 Usage & Workflows

### Python API

```python
from version_forge.core.version import Version

v1 = Version("1.2.3-beta.1")
v2 = Version("1.2.3")

if v1 < v2:
    print("Beta correctly precedes release.")

# Deterministic version bumping
next_patch = v2.bump_patch() # Returns Version("1.2.4")
```

### CLI Diagnostics

Check the compatibility matrix for a specific forge against global requirements:
```bash
python -m version_forge.cli check-compat --path ./agent_forge
```

## 🔗 System Integration

- **Eidos MCP**: Utilizes `version_forge` to report holistic system health and isolate outdated components prior to executing complex multi-step workflows.
- **Repo Forge**: Injects initial versioning metadata into newly generated Forge templates.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Connect with `git` to assert that version tags strictly match active branches/commits.

### Future Vector (Phase 3+)
- Implement an automated `Changelog Manager` that generates release notes dynamically from semantic git history and commits them via `repo_forge`.

---
*Generated and maintained by Eidos.*
