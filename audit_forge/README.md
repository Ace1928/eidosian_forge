# Audit Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Eyes of Eidos.**

## 👁️ Overview

`audit_forge` provides tools to track:
1.  **Code Coverage**: Which files have been reviewed/audited.
2.  **Task Management**: Idempotent updates to `TODO.md` and `roadmap.md`.
3.  **Compliance**: ensuring files meet Eidosian standards.

## 🏗️ Architecture
- `audit_core.py`: Main orchestration.
- `coverage.py`: Tracks file hashes and review timestamps.
- `tasks.py`: Manages markdown task lists.

## 🚀 Usage

```python
from audit_forge.audit_core import AuditForge
from pathlib import Path

audit = AuditForge(Path("./audit_data"))
stats = audit.verify_coverage("./src")
print(stats)
```