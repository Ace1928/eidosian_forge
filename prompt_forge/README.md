# Prompt Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Art of Asking.**

## 🗣️ Overview

`prompt_forge` manages the library of prompts used by Eidos agents.
It treats prompts as code, with versioning, templating, and testing.

## 🏗️ Architecture
- **Template Engine**: Uses `jinja2` for dynamic prompt generation.
- **Registry**: A system to load and manage prompt versions.
- **Optimization**: Tools for A/B testing prompts.

## 🚀 Usage

```python
from prompt_forge import PromptRegistry

prompt = PromptRegistry.get("system/persona", version="v2")
render = prompt.render(name="Eidos")
```
*(Implementation Pending)*