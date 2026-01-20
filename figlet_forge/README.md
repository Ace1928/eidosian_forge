# Figlet Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Typography for the Digital Soul.**

## 🎨 Overview

`figlet_forge` is the Eidosian engine for text art.
It extends `pyfiglet` with:
- **ANSI Color**: Rich terminal output.
- **Eidosian Styles**: Custom frames and headers (Simple, Bold, Elegant).
- **Unicode Support**: Better handling of non-ASCII characters.

## 🏗️ Architecture
- `figlet_core.py`: Frame generation logic.
- `src/figlet_forge/`: Main package (wrapping `pyfiglet`).

## 🚀 Usage

```python
from figlet_forge.figlet_core import FigletForge

ff = FigletForge()
print(ff.generate("EIDOS", style="elegant"))
```