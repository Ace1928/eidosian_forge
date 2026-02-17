# ⏳ Version Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Timekeeper of Eidos.**

> _"Stability requires a name; change requires a number."_

## ⏳ Overview

`version_forge` is the authority on semantic versioning and compatibility within the Eidosian ecosystem. It ensures that cross-forge dependencies are consistent and that the evolutionary history of the system is traceable.

## 🏗️ Architecture

- **SemVer Core (`version_core.py`)**: Implements strict SemVer 2.0 parsing, comparison, and increment logic.
- **Changelog Manager**: (Planned) Automated generation of `CHANGELOG.md` from git history and semantic commit messages.
- **Compatibility Matrix**: Validates component versions against system-wide requirements.

## 🔗 System Integration

- **Eidos MCP**: Uses version data to report system health and identify outdated components.
- **Repo Forge**: Injects initial versioning metadata into new Forge templates.

## 🚀 Usage

### Python API

```python
from version_forge.version_core import Version

v1 = Version("1.2.3-beta.1")
v2 = Version("1.2.3")

if v1 < v2:
    print("Beta is before release.")

# Bump version
next_patch = v2.bump_patch() # 1.2.4
```

### CLI

```bash
python -m version_forge.cli check-compat --path ./agent_forge
```
