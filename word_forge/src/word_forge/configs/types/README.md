# Word Forge Types Package

This package provides modularized type definitions for the Word Forge configuration system. It offers a cleaner, more organized alternative to importing from `config_essentials.py`.

## 📦 Module Structure

```
types/
├── __init__.py       # Package init with all re-exports
├── base.py           # Generic type variables and basic type aliases
├── errors.py         # Error handling (Error, Result, ErrorCategory)
├── workers.py        # Worker/task management types
├── protocols.py      # Protocol interfaces for components
├── templates.py      # Templates, TypedDicts, domain-specific types
├── enums.py          # Enumeration types
└── exceptions.py     # Exception classes
```

## 🚀 Quick Start

```python
# Import everything (backward compatible)
from word_forge.configs.types import *

# Import specific types (recommended)
from word_forge.configs.types import Error, Result, TaskPriority
from word_forge.configs.types import ConfigError, VectorConfigError
from word_forge.configs.types import StorageType, VectorModelType
```

## 📚 Module Reference

### base.py - Type Variables and Basic Types

Core type system building blocks:

```python
from word_forge.configs.types import (
    # Type variables
    T, R, K, V, E, C, T_contra,
    
    # Paths
    PROJECT_ROOT, DATA_ROOT, LOGS_ROOT,
    
    # JSON types
    JsonPrimitive, JsonDict, JsonList, JsonValue,
    
    # Config types
    ConfigValue, ConfigDict, PathLike, EnvVarType,
)
```

### errors.py - Error Handling

Monadic Result pattern for exception-free error handling:

```python
from word_forge.configs.types import Error, Result, ErrorCategory, ErrorSeverity

# Create a success result
result = Result.success(value)

# Create a failure result
result = Result.failure("ERR_CODE", "Something went wrong")

# Handle results
if result.is_success:
    value = result.unwrap()
else:
    print(f"Error: {result.error.message}")

# Chain operations
result.map(transform_fn).flat_map(async_operation)
```

### workers.py - Worker Management

Task and worker state management:

```python
from word_forge.configs.types import (
    TaskPriority,           # CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
    WorkerState,            # INITIALIZING, IDLE, PROCESSING, etc.
    CircuitBreakerState,    # CLOSED, OPEN, HALF_OPEN
    ExecutionMetrics,       # Performance measurement dataclass
    CircuitBreakerConfig,   # Circuit breaker settings
    TracingContext,         # Distributed tracing support
    measure_execution,      # Context manager for metrics
)

# Measure operation performance
with measure_execution("operation_name") as metrics:
    do_something()
print(f"Took {metrics.duration_ms}ms")
```

### protocols.py - Interfaces

Protocol definitions for component interfaces:

```python
from word_forge.configs.types import (
    ConfigComponent,       # Base protocol for config classes
    JSONSerializable,      # Protocol for JSON-serializable objects
    QueueProcessor,        # Protocol for queue processing
    WorkDistributor,       # Protocol for work distribution
    ConfigComponentInfo,   # Metadata about config components
)
```

### templates.py - Structured Types

TypedDicts and domain-specific types:

```python
from word_forge.configs.types import (
    # Templates
    InstructionTemplate, SQLitePragmas, SQLTemplates, TemplateDict,
    
    # Lexical types
    WordnetEntry, DictionaryEntry, DbnaryEntry, LexicalDataset,
    
    # Graph types
    WordTupleDict, RelationshipTupleDict, GraphInfoDict,
    
    # Worker pool config
    WorkerPoolConfig, TaskContext,
    
    # Literal types
    QueryType,                    # "search", "definition", "similarity"
    VectorDistanceMetric,         # "cosine", "euclidean", "dot", "manhattan"
    VectorSearchStrategy,         # "exact", "approximate", "hybrid"
    GraphExportFormat,            # "graphml", "gexf", "json", etc.
    TransactionIsolationLevel,    # Database isolation levels
)
```

### enums.py - Enumerations

All enum types with consistent string representation:

```python
from word_forge.configs.types import (
    StorageType,                  # MEMORY, DISK
    VectorModelType,              # TRANSFORMER, SENTENCE, WORD, CUSTOM
    VectorIndexStatus,            # UNINITIALIZED, READY, BUILDING, ERROR
    GraphLayoutAlgorithm,         # FORCE_DIRECTED, CIRCULAR, etc.
    GraphColorScheme,             # SEMANTIC, CATEGORY, SENTIMENT, etc.
    LogFormatTemplate,            # SIMPLE, STANDARD, DETAILED, JSON
    LogRotationStrategy,          # SIZE, TIME, NONE
    LogDestination,               # CONSOLE, FILE, BOTH, SYSLOG
    DatabaseDialect,              # SQLITE, POSTGRES, MYSQL, MEMORY
    QueuePerformanceProfile,      # LOW_LATENCY, HIGH_THROUGHPUT, etc.
    ConversationRetentionPolicy,  # KEEP_FOREVER, DELETE_AFTER_*, etc.
    ConversationExportFormat,     # JSON, MARKDOWN, TEXT, HTML
)
```

### exceptions.py - Exception Classes

Hierarchical exception types:

```python
from word_forge.configs.types import (
    # Config errors (inherit from ConfigError)
    ConfigError,
    PathError,
    EnvVarError,
    VectorConfigError,
    VectorIndexError,
    GraphConfigError,
    LoggingConfigError,
    DatabaseConfigError,
    DatabaseConnectionError,
    
    # Lexical resource errors
    LexicalResourceError,
    ResourceNotFoundError,
    ResourceParsingError,
    
    # Model errors
    ModelError,
    
    # Worker errors (inherit from WorkerError)
    WorkerError,
    TaskExecutionError,
    QueueOperationError,
    CircuitOpenError,
    EmptyQueueError,
)
```

## 🔄 Backward Compatibility

For full backward compatibility, you can still import from `config_essentials.py`:

```python
# Legacy import (still works)
from word_forge.configs.config_essentials import Error, Result, TaskPriority

# New modular import (recommended)
from word_forge.configs.types import Error, Result, TaskPriority
```

Both approaches provide identical types - use whichever suits your codebase.

## 🏗️ Architecture

The types package follows Eidosian principles:

1. **Modular Design**: Each module has a single responsibility
2. **Type Safety**: Comprehensive type hints throughout
3. **Backward Compatible**: All existing imports continue to work
4. **Self-Documenting**: Rich docstrings with examples
5. **Consistent Representation**: All enums use `EnumWithRepr` base

---

*Part of the Eidosian Forge ecosystem*
