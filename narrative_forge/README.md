# 📖 Narrative Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Storyteller of Eidos.**

> _"We are the stories we tell ourselves."_

## 📖 Overview

`narrative_forge` is the engine for long-form coherence and storytelling. Unlike `agent_forge`, which focuses on immediate execution cycles (`beats`), `narrative_forge` maintains the "arc" of interaction, ensuring thematic consistency over time.

## 🏗️ Architecture

- **Engine (`engine.py`)**: Manages the narrative state machine.
- **Memory (`memory.py`)**: A dedicated context window for story beats (distinct from `memory_forge`'s vector store).
- **CLI (`cli.py`)**: Interactive storytelling interface.

## 🔗 System Integration

- **Agent Forge**: Agents can query the narrative engine to understand "where they are" in the larger plot.
- **Word Forge**: Uses the semantic lexicon to enrich descriptions.

## 🚀 Usage

```bash
# Start an interactive narrative session
python -m narrative_forge.cli chat --model "local"
```
