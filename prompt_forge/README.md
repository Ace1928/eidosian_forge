# 🗣️ Prompt Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Art of Asking.**

> _"The quality of the thought depends on the clarity of the prompt."_

## 🗣️ Overview

`prompt_forge` is the centralized repository for Eidosian alignment strategies. It manages the "Source of Truth" for agent personas, system instructions, and few-shot examples.

## 🏗️ Architecture

- **Asset Library**: A curated collection of Markdown and text-based prompt templates.
- **Strategic Alignment**: Standardized personas (e.g., "Velvet Beef") ensuring consistent behavior across different models.
- **Registry**: (In Development) A Python API to load and version prompts dynamically.

## 🔗 System Integration

- **Eidos MCP**: Uses templates from `prompt_forge` to prime the Documentation and Refactor agents.
- **Agent Forge**: Provides the core identity and directive sets for the daemon loop.

## 🚀 Usage

Currently, prompts are consumed as static files.

```bash
# Example: Using a persona template
cat prompt_forge/github_copilot_prompt_guide.md
```

## 📂 Notable Templates

- `eidos_persona.md`: Core identity definition.
- `documentation_agent.txt`: Instructions for the Documentation Forge.
- `refactor_instruction.txt`: Rules for structural code modification.
