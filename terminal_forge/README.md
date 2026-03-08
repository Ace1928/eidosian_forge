# 📟 Terminal Forge ⚡

> _"The Face of Eidos. High-fidelity visual communication at the command line."_

## 🧠 Overview

`terminal_forge` is the visual presentation layer for Eidosian interfaces. It provides a standardized suite of Terminal User Interface (TUI) widgets, layouts, and strictly typed color palettes built heavily upon `rich` (and eventually `textual` for interactivity).

```ascii
      ╭───────────────────────────────────────────╮
      │             TERMINAL FORGE                │
      │    < Rich Themes | TUI Layouts | ANSI >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   RICH RENDERERS    │   │  ASCII WIDGETS  │
      │ (Console Formatting)│   │ (Borders/Panels)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: UI / UX Components
- **Test Coverage**: UI logic and theme generation verified.
- **Architecture**:
  - `rich_theme.py`: Standardized Eidosian `rich.theme.Theme` generators.
  - `ascii_art.py` / `banner.py`: Display primitives mapping closely to `figlet_forge`.
  - `layout.py`: Structured panel alignments.

## 🚀 Usage & Workflows

### Python API

```python
from rich.console import Console
from terminal_forge import build_eidosian_rich_theme

# Initialize an Eidosian-styled console
console = Console(theme=build_eidosian_rich_theme("dark"))
console.print("Eidosian terminal theme active", style="panel.title")
```

## 🔗 System Integration

- **All Forges**: Used uniformly by all CLI entry points across the `*_forge` ecosystem to guarantee a consistent visual identity (The Eidosian Aesthetic).
- **Diagnostics Forge**: Used to format structured error logs and operational traces.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Migrate any remaining `print()` statements in legacy forges to use `terminal_forge.Console`.

### Future Vector (Phase 3+)
- Add explicit Textual widgets for rapid scaffolding of complex interactive dashboards (e.g., merging `moltbook_forge` HTML UI into a pure TUI equivalent).

---
*Generated and maintained by Eidos.*
