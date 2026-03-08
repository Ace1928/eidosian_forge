# 🔪 Refactor Forge ⚡

> _"The Scalpel of Eidos. Precision modification without loss of intent."_

## 🧠 Overview

`refactor_forge` provides structural, AST-based code transformation capabilities. It allows the Eidosian runtime to safely analyze, manipulate, and optimize Python codebases programmatically without breaking semantic structures.

```ascii
      ╭───────────────────────────────────────────╮
      │             REFACTOR FORGE                │
      │    < Parse | Transform | Unparse >        │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   AST ANALYSIS      │   │  DIFF VIEWER    │
      │ (Structure Map)     │   │ (Preview Edits) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Code Manipulation & Transformation
- **Test Coverage**: 100% Core AST manipulations tested.
- **Architecture**:
  - `analyzer.py`: AST-based metrics and boundary detection.
  - `diff_viewer.py`: Rich terminal outputs for proposed changes.
  - `cli.py`: Typer-based CLI for analysis and refactoring previews.

## 🚀 Usage & Workflows

### Python API

```python
from refactor_forge import RefactorForge

rf = RefactorForge()

# Safely rename identifiers via AST (avoiding blind regex replacement)
transformed_code = rf.transform(
    source="def old_func(): pass",
    rename_map={"old_func": "new_func"},
    remove_docs=True
)
print(transformed_code)
# Output: def new_func():\n    pass
```

### CLI Diagnostics

Analyze code boundaries and structures:
```bash
python -m refactor_forge.cli path/to/source.py --analyze-only
```

Preview differences:
```bash
python -m refactor_forge.cli path/to/source.py --dry-run --preview-diff-against path/to/proposed.py
```

## 🔗 System Integration

- **Eidos MCP**: Exposes `refactor_analyze` to the cognitive layer.
- **Agent Forge**: Agents leverage the AST analysis to understand dependencies before modifying complex files.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Add explicit support for unparsing ASTs in Python < 3.9 (or enforce 3.9+).

### Future Vector (Phase 3+)
- Integrate with `code_forge` digester to allow automated application of reduction candidates directly to the filesystem.

---
*Generated and maintained by Eidos.*
