# Word Forge Developer Guide

This guide provides a comprehensive overview of Word Forge's architecture, components, and development practices.

## Architecture Overview

Word Forge follows a modular architecture with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI (forge.py)                        │
│                     Entry point for users                    │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                    Configuration System                       │
│            config.py + configs/config_essentials.py          │
│         Centralized settings with environment overrides       │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                       Core Modules                           │
├──────────┬───────────┬──────────┬────────────┬──────────────┤
│ database │   graph   │ emotion  │ vectorizer │    parser    │
│ (SQLite) │(NetworkX) │ (VADER)  │ (ST/Chroma)│   (NLTK)     │
└──────────┴───────────┴──────────┴────────────┴──────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                    Queue & Workers                           │
│            Threaded background processing system             │
└──────────────────────────────────────────────────────────────┘
```

## Core Modules

### database

Manages SQLite storage for lexical data including words, definitions, relationships, and emotional associations.

**Key Classes:**
- `DBManager` - Core database operations with connection pooling
- `DatabaseWorker` - Background worker for database maintenance

**Features:**
- Connection pooling for efficient resource usage
- Transaction management with automatic rollback
- Schema migrations support
- Relationship type management

### emotion

Computes emotional metrics for words and passages using multiple analysis methods.

**Key Classes:**
- `EmotionManager` - Unified interface for emotional processing
- `EmotionProcessor` - Low-level emotion computation
- `EmotionWorker` - Background annotation worker

**Features:**
- Dimensional analysis (valence/arousal)
- Categorical classification (happiness, sadness, anger, etc.)
- Multi-method sentiment fusion (TextBlob, VADER, LLM)
- Emotional relationship tracking

### graph

Builds and manages a semantic network from processed terms using NetworkX.

**Key Classes:**
- `GraphManager` - Central orchestrator for graph operations
- `GraphBuilder` - Constructs graph from database
- `GraphWorker` - Background graph maintenance
- `GraphAnalysis` - Graph metrics and insights
- `GraphVisualizer` - 2D/3D visualization generation

**Features:**
- Multi-dimensional relationship support
- Layout algorithms (force-directed, hierarchical, spectral)
- Export formats (GraphML, GEXF, JSON, HTML)
- Community detection and clustering

### parser

Extracts and refines lexical entries from text sources.

**Key Classes:**
- `ParserRefiner` - Main parsing pipeline
- `TermExtractor` - NLP-based term discovery
- `LexicalResources` - Resource path management

**Features:**
- WordNet integration
- Thesaurus data processing
- Language model enhancement
- Relationship extraction

### vectorizer

Generates and manages vector embeddings for semantic search.

**Key Classes:**
- `VectorStore` - Embedding storage and retrieval
- `VectorWorker` - Background indexing worker

**Features:**
- Multiple backend support (ChromaDB, FAISS, SQLite)
- Sentence transformer models
- Hybrid search strategies
- Configurable distance metrics

### queue

Orchestrates worker tasks for asynchronous processing.

**Key Classes:**
- `QueueManager` - Thread-safe task queue
- `WorkerManager` - Worker lifecycle management
- `WordProcessor` - Lexical item processing

**Features:**
- Priority-based scheduling
- Result monad pattern for error handling
- Performance metrics collection
- Backpressure management

## Configuration System

The configuration system (`config.py` + `configs/config_essentials.py`) provides:

- **Type-safe settings** with dataclass-based components
- **Environment variable overrides** for deployment flexibility
- **Runtime configuration changes** with observer notifications
- **Validation** for all configuration values
- **Serialization** for export and debugging

### Configuration Components

| Component | Purpose | Key Settings |
|-----------|---------|--------------|
| `DatabaseConfig` | Database connection | `db_path`, `pool_size`, `pragmas` |
| `VectorizerConfig` | Vector operations | `model_name`, `batch_size`, `storage_type` |
| `ParserConfig` | Text parsing | `data_dir`, `preload_resources` |
| `EmotionConfig` | Emotion analysis | `analyzer_weights`, `threshold` |
| `GraphConfig` | Graph operations | `layout_algorithm`, `export_format` |
| `QueueConfig` | Queue behavior | `max_workers`, `queue_size` |
| `LoggingConfig` | Logging setup | `level`, `file_path`, `format` |

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/Ace1928/word_forge.git
cd word_forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type Checking**: mypy (optional)

```bash
# Format code
black .

# Check linting
ruff check .

# Auto-fix issues
ruff check . --fix

# Type checking
mypy src/word_forge
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=word_forge --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run tests matching pattern
pytest -k "test_emotion"
```

### Adding New Features

1. Create feature module in appropriate package
2. Add configuration if needed in `configs/`
3. Write tests in `tests/`
4. Update documentation in `docs/`
5. Add to `__all__` exports if public API

## Additional Resources

- [`DETAILED_TODO.md`](../DETAILED_TODO.md) - Comprehensive improvement roadmap (primary reference)
- [`docs/glossary.md`](glossary.md) - Term definitions
- [`docs/templates/`](templates/) - Docstring templates for new code
- [`upgrade_plan.md`](../upgrade_plan.md) - Architectural analysis (reference only)
