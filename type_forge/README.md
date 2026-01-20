# Type Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Laws of Eidos.**

## 📏 Overview

`type_forge` provides runtime data validation.
It implements a lightweight schema engine (similar to JSON Schema) for validating dynamic data structures passed between agents.

## 🏗️ Architecture
- `type_core.py`: Main validation engine.

## 🚀 Usage

```python
from type_forge.type_core import TypeCore

tc = TypeCore()
schema = {"type": "string", "minLength": 5}
tc.check_schema(schema, "Hello World") # True
```