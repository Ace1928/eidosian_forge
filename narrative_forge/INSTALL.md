# 📖 Narrative Forge Installation

Story generation, character development, and narrative structure tools.

## Quick Install

```bash
pip install -e ./narrative_forge

# Verify
python -c "from narrative_forge import StoryEngine; print('✓ Ready')"
```

## CLI Usage

```bash
# Generate story outline
narrative-forge outline "A hero's journey" --style epic

# Create character
narrative-forge character create --name "Alice" --archetype hero

# Help
narrative-forge --help
```

## Python API

```python
from narrative_forge import StoryEngine, Character

# Create story
story = StoryEngine()
story.add_character(Character("Alice", archetype="hero"))
outline = story.generate_outline()
```

## Dependencies

- `eidosian_core` - Universal decorators and logging
- LLM integration for generation (via llm_forge)

