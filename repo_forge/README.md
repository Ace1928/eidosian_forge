# 📐 Repo Forge ⚡

> _"The Blueprint of Eidos. Standards are the scaffolding of emergence."_

## 🧠 Overview

`repo_forge` is the structural custodian of the Eidosian ecosystem. It ensures that every forge, module, and script adheres to the strict organizational protocols required for seamless integration, automated testing, and recursive documentation.

```ascii
      ╭───────────────────────────────────────────╮
      │               REPO FORGE                  │
      │    < Scaffold | Diagnostics | Registry >  │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   FORGE GENERATOR   │   │ SYSTEM ATLAS    │
      │ (Standard Layouts)  │   │ (Manifest Sync) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Infrastructure & Project Generation
- **Test Coverage**: Generator scaffolding verified.
- **Architecture**:
  - `generators/`: Standardized Eidosian templates for new Forges.
  - `core/diagnostics.py`: Scans repository health (missing READMEs, bad imports).
  - `core/registry.py`: Centralized manifest of all active Forges.
  - `cli.py`: Typer-based command interface.

## 🚀 Usage & Workflows

### Generating a New Forge

Scaffold a completely compliant new Forge, ready for Eidosian integration:
```bash
python -m repo_forge.cli create my_new_forge
```

### Diagnostics

Run the "doctor" command to verify that all current Forges meet minimum structural requirements:
```bash
python -m repo_forge.cli doctor
```

## 🔗 System Integration

- **Documentation Forge**: `doc_forge` consumes the `repo_forge` registry to map out its automatic documentation sweeps.
- **Eidos MCP**: Inherits system-wide structural awareness natively.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Add pre-commit hook enforcement validation to `doctor` command.

### Future Vector (Phase 3+)
- Integrate with `code_forge` digester to auto-generate pull requests fixing structural drift (e.g., auto-scaffolding missing test directories).

---
*Generated and maintained by Eidos.*
