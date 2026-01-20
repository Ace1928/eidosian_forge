# Test Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Crucible of Eidos.**

## 🧪 Overview

`test_forge` provides shared test fixtures, mocks, and utilities for the Eidosian ecosystem.
It ensures that all Forges meet the rigorous standards defined in `global_info.py`.

## 🏗️ Architecture
- `src/test_forge/`: Core utilities.

## 🚀 Usage

```python
from test_forge import EidosFixture

def test_agent(eidos: EidosFixture):
    assert eidos.is_alive()
```