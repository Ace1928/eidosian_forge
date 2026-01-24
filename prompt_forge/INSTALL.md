# 💬 Prompt Forge Installation

LLM prompt engineering, templates, and optimization.

## Quick Install

```bash
pip install -e ./prompt_forge

# Verify
python -c "from prompt_forge import PromptTemplate; print('✓ Ready')"
```

## CLI Usage

```bash
# Create prompt from template
prompt-forge render template.md --vars '{"name": "Alice"}'

# Optimize prompt
prompt-forge optimize "Explain {topic}" --model gpt-4

# Help
prompt-forge --help
```

## Python API

```python
from prompt_forge import PromptTemplate, PromptOptimizer

# Create template
template = PromptTemplate("Explain {topic} in {style} terms")
prompt = template.render(topic="quantum physics", style="simple")
```

## Dependencies

- `jinja2` - Template rendering
- `eidosian_core` - Universal decorators and logging

