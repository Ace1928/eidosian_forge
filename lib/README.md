# ⚛️ Eidosian Core (`lib`)

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Fundamental Primitives of Eidos.**

> _"In the beginning, there was the Decorator."_

## 🧠 Overview

The `lib` directory contains `eidosian_core`, the shared library that binds all Forges together. It provides the runtime substrate for logging, tracing, configuration, and the unified Eidosian identity.

## 🏗️ Core Features

### 1. The `@eidosian` Decorator
A universal wrapper that provides:
- **Tracing**: Automatic span generation for observability.
- **Logging**: Structured, context-aware logging.
- **Error Handling**: Standardized exception wrapping.

```python
from eidosian_core import eidosian

@eidosian(scope="agent_core")
def think(context):
    pass
```

### 2. Global Information System (GIS)
A unified configuration interface backed by `config.json` and environment variables.

### 3. Profiling & Benchmarking
Utilities to measure the cognitive and computational cost of operations.

## 🔗 System Integration

Every Forge depends on `eidosian_core`. It is the only hard dependency across the entire ecosystem.

## 🚀 Installation

```bash
pip install -e lib
```
