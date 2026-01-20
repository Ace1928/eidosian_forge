# Version Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Timekeeper of Eidos.**

## ⏳ Overview

`version_forge` manages semantic versioning and compatibility checks for Eidosian components.
It implements:
- **SemVer 2.0**: Parsing and comparison.
- **Dependency Resolution**: Basic constraint checking (`^`, `~`, `==`).

## 🏗️ Architecture
- `version_core.py`: SemVer implementation.

## 🚀 Usage

```python
from version_forge.version_core import Version, VersionForge

v1 = Version("1.2.3")
v2 = Version("2.0.0")
assert v1 < v2
```