# word_forge Package

This package contains the core modules of the Word Forge project.

## Package Structure

```
word_forge/
├── config.py              # Central configuration management
├── forge.py               # CLI entry point and orchestration
├── exceptions.py          # Custom exception hierarchy
├── relationships.py       # Relationship type definitions
├── configs/               # Configuration component modules
│   ├── config_essentials.py  # Type definitions and utilities
│   └── logging_config.py     # Logging configuration
├── database/              # Persistent lexical storage
│   ├── database_manager.py   # Core database operations
│   ├── database_config.py    # Database configuration
│   └── database_worker.py    # Background maintenance
├── emotion/               # Emotional analysis system
│   ├── emotion_manager.py    # Unified emotion interface
│   ├── emotion_processor.py  # Low-level processing
│   ├── emotion_config.py     # Emotion configuration
│   ├── emotion_types.py      # Type definitions
│   └── emotion_worker.py     # Background annotation
├── graph/                 # Semantic network operations
│   ├── graph_manager.py      # Central orchestrator
│   ├── graph_builder.py      # Graph construction
│   ├── graph_analysis.py     # Metrics and insights
│   ├── graph_visualizer.py   # Visualization generation
│   ├── graph_worker.py       # Background updates
│   └── graph_config.py       # Graph configuration
├── parser/                # Text parsing and extraction
│   ├── parser_refiner.py     # Main parsing pipeline
│   ├── lexical_functions.py  # Lexical data functions
│   ├── language_model.py     # LLM integration
│   └── parser_config.py      # Parser configuration
├── vectorizer/            # Vector embedding operations
│   ├── vector_store.py       # Embedding storage/search
│   ├── vector_worker.py      # Background indexing
│   └── vectorizer_config.py  # Vector configuration
├── queue/                 # Worker coordination
│   ├── queue_manager.py      # Thread-safe task queue
│   ├── queue_worker.py       # Word processor
│   ├── worker_manager.py     # Worker lifecycle
│   └── queue_config.py       # Queue configuration
├── conversation/          # Conversation management
│   ├── conversation_manager.py  # Session orchestration
│   ├── conversation_models.py   # Data models
│   └── conversation_worker.py   # Background processing
├── utils/                 # Shared utilities
│   └── nltk_utils.py         # NLTK data management
├── demos/                 # Example scripts
│   ├── config_demo.py
│   ├── database_demo.py
│   └── graph_demo.py
└── tools/                 # Additional tools
    └── av_to_text.py         # Audio/video transcription
```

## Design Principles

All modules follow the **Eidosian** pattern:

1. **Type Safety** - Comprehensive type hints and runtime validation
2. **Clear Layering** - Each module has a single responsibility
3. **Self-Documenting** - Thorough docstrings with examples
4. **Functional Design** - Pure functions where possible
5. **Testability** - Dependency injection for easy mocking

## Usage

```python
# Import the central configuration
from word_forge.config import config

# Access specific modules
from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.vectorizer.vector_store import VectorStore
```

## Dependencies

Core dependencies are managed in `pyproject.toml`. Optional heavy dependencies (torch, transformers, etc.) are guarded with try/except imports to allow the package to function in lightweight environments.
