# 🎨 Figlet Forge ⚡

> _"Typography for the Digital Soul. Shaping the visual identity of the Eidosian runtime."_

## 🧠 Overview

`figlet_forge` is the Eidosian engine for text art and terminal typography. It acts as an advanced wrapper and extension of `pyfiglet`, providing a strict, stylized visual interface for CLI outputs and structural logs.

```ascii
      ╭───────────────────────────────────────────╮
      │               FIGLET FORGE                │
      │    < Typography | Color | Formatting >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   ASCII FRAMES      │   │   ANSI STYLES   │
      │ (Structural Art)    │   │ (Rich Output)   │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Visual & Output Formatting
- **Test Coverage**: Extensive integration and compat suites passing (562+ tests).
- **Core Capabilities**:
  - **ANSI Color**: Rich terminal output.
  - **Eidosian Styles**: Custom frames and headers (Simple, Bold, Elegant).
  - **Unicode Support**: Extended handling of non-ASCII boundaries.

## 🚀 Usage & Workflows

### Python API

```python
from figlet_forge.figlet_core import FigletForge

# Initialize the Eidosian typography engine
ff = FigletForge()

# Generate styled output
banner = ff.generate("EIDOS", style="elegant")
print(banner)
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Ensure test suite stability across environments (Termux confirmed).

### Future Vector
- Integrate deeply with `glyph_forge` to establish a unified cross-platform UX schema.
- Develop dynamic, context-aware styling (e.g., coloring based on sentiment or error severity from `diagnostics_forge`).

---
*Generated and maintained by Eidos.*
