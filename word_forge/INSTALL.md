# Word Forge Installation Guide

## Quick Install

```bash
# From the word_forge directory
pip install -e .

# Or install with all extras
pip install -e ".[full]"
```

## Recommended Environment

Use the shared Eidosian virtual environment when running in this repo:

```bash
/home/lloyd/eidosian_forge/eidosian_venv/bin/python -m pip install -e .
```

## Prerequisites

Word Forge uses NLTK for WordNet access. Download required data:
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Local Ollama Models (Optional)

To run fully local:

```bash
ollama pull qwen2.5:1.5b-Instruct
ollama pull nomic-embed-text
```

Then pass model names with the `ollama:` prefix in CLI or daemon config.

## Verify Installation

```bash
# Check CLI is available
word-forge --version

# Check system status
word-forge status
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/word_forge/completions/word-forge.bash

# Or for a one-time session
source completions/word-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
Word Forge works independently with these capabilities:
- WordNet lookups and definitions
- Synset exploration
- Related word discovery
- Semantic graph building

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `knowledge_forge` | Cross-link lexical concepts |
| `llm_forge` | AI-enhanced definitions |

Check available integrations:
```bash
word-forge integrations
```

## CLI Commands

```bash
word-forge status              # Show lexicon status
word-forge lookup <word>       # Look up word in lexicon
word-forge define <word>       # Get word definition
word-forge related <word>      # Find related words
word-forge synsets <word>      # Get WordNet synsets
word-forge graph               # Show graph statistics
word-forge build <word>        # Build lexical entry
```

## Python API

```python
from word_forge import Config, GraphManager, GraphBuilder, GraphQuery
from word_forge.parser.lexical_functions import (
    create_lexical_dataset,
    get_wordnet_data,
    get_synsets,
)

# Get word data
data = get_wordnet_data("intelligence")

# Get synsets
synsets = get_synsets("intelligence")
for syn in synsets:
    print(f"{syn.name()}: {syn.definition()}")

# Build lexical dataset
dataset = create_lexical_dataset("philosophy")
```

## Data Locations

| Data | Location |
|------|----------|
| Database | `~/.word_forge/word_forge.db` |
| Graph | `~/.word_forge/graph.pkl` |
| Config | `~/.word_forge/config.yaml` |

## Troubleshooting

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('wordnet')"
```

### Import Errors
Ensure the lib directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### Graph Initialization Slow
The first graph initialization may take time. This is normal.
