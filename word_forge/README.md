# Word Forge

[![CI](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Word Forge is a modular lexical processing and enrichment toolkit that builds a comprehensive semantic network while providing vector search, emotion analysis, and graph visualization capabilities. The project embraces the **Eidosian** design philosophy—typed interfaces, clear separation of concerns, and recursive self‑improvement.

## Features

- **Lexical Data Aggregation**: Combines data from WordNet, thesauruses, and other linguistic resources
- **Semantic Graph**: Builds a multidimensional knowledge graph using NetworkX with relationship types including synonyms, antonyms, hypernyms, and emotional associations
- **Emotion Analysis**: Dimensional (valence/arousal) and categorical emotion detection using VADER, TextBlob, and optional LLM integration
- **Vector Search**: Semantic similarity search powered by sentence transformers and ChromaDB/FAISS backends
- **CLI Interface**: Full-featured command-line interface for all major operations
- **Background Workers**: Threaded workers for graph building, vector indexing, and emotion annotation

## Architecture

```
┌────────────────────┐
│    word_forge      │  CLI entry point
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│       config       │  Centralized configuration
└─────────┬──────────┘
          │
┌─────────┴─────────────────────────────────────────┐
│                    Core Modules                    │
├──────────┬───────────┬──────────┬────────────────┤
│ database │   graph   │ emotion  │   vectorizer   │
│(SQLite)  │(NetworkX) │(VADER/TB)│ (Transformers) │
└──────────┴───────────┴──────────┴────────────────┘
          │
┌─────────▼──────────┐
│   queue/workers    │  Background processing
└────────────────────┘
```

## Installation

Word Forge targets **Python 3.8 or newer**.

### Basic Installation

```bash
# Clone and install locally
git clone https://github.com/Ace1928/word_forge.git
cd word_forge
pip install -e .

# Or install directly from Git
pip install git+https://github.com/Ace1928/word_forge.git
```

### Development Installation

Install with development tools (formatter, linter, tests):

```bash
pip install -e .[dev]
```

### Optional Feature Extras

Install feature bundles based on your needs:

| Extra | Command | Includes |
|-------|---------|----------|
| `vector` | `pip install -e .[vector]` | sentence-transformers, ChromaDB, FAISS |
| `visualization` | `pip install -e .[visualization]` | Pyvis, Plotly |
| `dev` | `pip install -e .[dev]` | black, ruff, pytest, mypy, pre-commit |

**Note**: The `word_forge` CLI relies on the `vector` extra for semantic search operations.

## Quick Start

### Python API

```python
from word_forge.config import config
from word_forge.database.database_manager import DBManager

# Initialize database
db = DBManager()
db.create_tables()

# Add a word entry
db.insert_or_update_word(
    term="algorithm",
    definition="A step-by-step procedure for solving a problem",
    part_of_speech="noun"
)

# Create relationships
db.insert_relationship("algorithm", "procedure", "synonym")
```

### Command Line Interface

The package installs a `word_forge` executable:

```bash
# Show version
word_forge --version

# Start the processing pipeline with seed words
word_forge start apple banana --minutes 5 --workers 4

# Use custom database, vector, and LLM models (useful for isolating runs)
word_forge start apple --db-path /tmp/word_forge.sqlite \
  --vector-model sentence-transformers/all-MiniLM-L6-v2 \
  --llm-model sshleifer/tiny-gpt2

# Build the semantic graph
word_forge graph build --timeout 180

# Generate visualization (requires visualization extra)
word_forge graph visualize --3d --open-browser

# Index vectors
word_forge vector index --embedder sentence-transformers/all-MiniLM-L6-v2

# Search for similar terms
word_forge vector search "happy" --top-k 10

# Start a new conversation
word_forge conversation start --title "My Session"

# List conversations
word_forge conversation list --limit 5

# Show messages in a conversation
word_forge conversation show 1 --limit 20

# Annotate emotions
word_forge emotion annotate --strategy hybrid

# Run full demo pipeline
word_forge demo full --3d --open-browser

# Setup NLTK data
word_forge setup-nltk

# Quiet mode (suppress non-error output)
word_forge --quiet start apple

# Verbose mode (enable debug output)
word_forge --verbose start apple
```

Note: The first run may download NLTK corpora and sentence-transformer models.

### Demo Scripts

```bash
# Explore configuration
python -m word_forge.demos.config_demo --validate

# Vector worker demo
python -m word_forge.demos.vector_worker_demo

# Generate lexical data for a word
python - <<'PY'
from word_forge.parser.lexical_functions import create_lexical_dataset

print(create_lexical_dataset("recursion"))
PY

# Database demo
python -m word_forge.demos.database_demo
```

## Project Structure

```
word_forge/
├── src/word_forge/        # Main package
│   ├── config.py          # Central configuration
│   ├── forge.py           # CLI entry point
│   ├── configs/           # Configuration components
│   ├── database/          # SQLite persistence layer
│   ├── emotion/           # Emotion analysis system
│   ├── graph/             # Semantic graph operations
│   ├── parser/            # Text parsing and lexical extraction
│   ├── queue/             # Worker queue management
│   ├── vectorizer/        # Vector embeddings and search
│   └── demos/             # Example scripts
├── tests/                 # Test suite (pytest)
├── docs/                  # Documentation
├── data/                  # Data directory (created at runtime)
└── pyproject.toml         # Project configuration
```

## Development

### Code Style

Word Forge uses **Black** for formatting and **Ruff** for linting:

```bash
# Format code
black .

# Lint code
ruff check .

# Auto-fix lint issues
ruff check . --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=word_forge --cov-report=html
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint on commit:

```bash
pip install pre-commit
pre-commit install
```

## Configuration

Word Forge uses a centralized configuration system with environment variable overrides:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `WORDFORGE_DB_PATH` | Database file path | `data/word_forge.sqlite` |
| `WORDFORGE_LOG_LEVEL` | Logging level | `INFO` |
| `WORDFORGE_VECTOR_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |

Configuration can also be modified programmatically:

```python
from word_forge.config import config

# Access configuration
print(config.database.db_path)
print(config.vectorizer.model_name)

# Export configuration
config.export_to_file("config.json")
```

## NLTK Data

Word Forge uses NLTK for WordNet and other linguistic resources. Data is downloaded automatically on first use, or can be pre-downloaded:

```bash
word_forge setup-nltk
```

Required corpora: WordNet, Punkt, stopwords, VADER lexicon.

## Troubleshooting

### Common Issues

#### NLTK Data Not Found

If you encounter errors about missing NLTK corpora:

```bash
# Download required NLTK data
word_forge setup-nltk

# Or manually in Python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

#### Memory Issues with Large Models

For systems with limited RAM:

1. Use a smaller embedding model:
   ```bash
   export WORDFORGE_VECTOR_MODEL="sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
   ```

2. Reduce batch sizes in configuration:
   ```python
   from word_forge.config import config
   config.vectorizer.batch_size = 16
   ```

3. Use the low-memory profile:
   ```bash
   python -m word_forge.demos.config_demo --profile low_memory
   ```

#### ChromaDB/FAISS Import Errors

These heavy dependencies are optional. Install them explicitly:

```bash
pip install -e .[vector]
```

#### SQLite Database Locked

If you see "database is locked" errors:

1. Ensure only one process writes to the database at a time
2. Check for zombie processes: `ps aux | grep word_forge`
3. Use WAL mode (enabled by default) for better concurrency

#### Tests Failing with Import Errors

CI uses a lightweight test configuration. For full tests locally:

```bash
pip install -e .[dev]
pytest
```

### Getting Help

- Check [docs/overview.md](docs/overview.md) for detailed component documentation
- Review [docs/glossary.md](docs/glossary.md) for term definitions
- Open an issue on GitHub for bugs or feature requests

## API Quick Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `DBManager` | `database.database_manager` | SQLite database operations |
| `GraphManager` | `graph.graph_manager` | Semantic graph construction |
| `EmotionManager` | `emotion.emotion_manager` | Emotion analysis |
| `VectorStore` | `vectorizer.vector_store` | Vector embeddings & search |
| `QueueManager` | `queue.queue_manager` | Task queue management |
| `ParserRefiner` | `parser.parser_refiner` | Text parsing pipeline |
| `ConversationManager` | `conversation.conversation_manager` | Multi-turn conversations |

### Common Operations

```python
# Database operations
from word_forge.database.database_manager import DBManager
db = DBManager()
db.create_tables()
db.insert_or_update_word("example", "a representative sample", "noun")
entry = db.get_word_entry("example")

# Graph operations
from word_forge.graph.graph_manager import GraphManager
graph = GraphManager(db_manager=db)
graph.build_graph()
graph.visualize(output_path="graph.html")

# Emotion analysis
from word_forge.emotion.emotion_manager import EmotionManager
em = EmotionManager(db)
valence, arousal = em.analyze_text_emotion("I love this!")

# Vector search
from word_forge.vectorizer.vector_store import VectorStore
vs = VectorStore(db_manager=db)
results = vs.search(query_text="happy", k=5)
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `word_forge start [WORDS...]` | Start processing pipeline |
| `word_forge graph build` | Build semantic graph |
| `word_forge graph visualize` | Generate visualization |
| `word_forge vector index` | Index word vectors |
| `word_forge vector search QUERY` | Search for similar terms |
| `word_forge conversation start` | Start new conversation |
| `word_forge conversation list` | List conversations |
| `word_forge conversation show ID` | Show messages in a conversation |
| `word_forge emotion annotate` | Run emotion annotation |
| `word_forge demo full` | Run full demo pipeline |
| `word_forge setup-nltk` | Download NLTK data |

## Documentation

- [`docs/overview.md`](docs/overview.md) - Developer guide
- [`docs/glossary.md`](docs/glossary.md) - Term definitions
- [`docs/templates/`](docs/templates/) - Docstring templates

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure tests pass (`pytest`)
5. Ensure code is formatted (`black . && ruff check .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

All pull requests must pass CI checks before merging.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NetworkX](https://networkx.org/) - Graph operations
- [NLTK](https://www.nltk.org/) - Natural language processing
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
