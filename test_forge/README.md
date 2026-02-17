# 🧪 Test Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Crucible of Eidos.**

> _"Trust, but verify. Then verify the verification."_

## 🧪 Overview

`test_forge` provides the shared testing infrastructure for the Eidosian ecosystem. It standardizes fixtures, mocks, and validation logic to ensure that every Forge adheres to the Eidosian quality standards.

## 🏗️ Architecture

- **Fixtures (`fixtures.py`)**: Standard `pytest` fixtures for mocking `eidos_mcp`, `memory_forge`, and `ollama_forge`.
- **Validators (`validators.py`)**: Utilities to check structural integrity of outputs (JSON schemas, Markdown formats).
- **Benchmarks (`benchmarks/`)**: Performance regression tests.

## 🔗 System Integration

All Forges use `test_forge` in their `dev` dependencies.

## 🚀 Usage

### Using Fixtures

```python
import pytest
from test_forge import mock_nexus

def test_agent_reasoning(mock_nexus):
    mock_nexus.expect_tool_call("memory_search")
    # ... run agent ...
    assert mock_nexus.tool_called("memory_search")
```

### Running Tests

```bash
pytest --cov=src
```
