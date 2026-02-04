"""
Word Forge Configuration Types Package.

This package provides modularized type definitions for the Word Forge
configuration system. All types are re-exported from this package for
backward compatibility.

Modules:
    base: Generic type variables and basic type aliases
    errors: Error handling types (Error, Result, ErrorCategory)
    workers: Task and worker management types
    protocols: Protocol interfaces for configuration components
    templates: Template, schema, and domain-specific type definitions
    enums: Enumeration types for configuration options
    exceptions: Exception types for error handling
"""

# Re-export everything for backward compatibility
from .base import (
    C,
    K,
    V,
    E,
    R,
    T,
    T_contra,
    DATA_ROOT,
    LOGS_ROOT,
    PROJECT_ROOT,
    ComponentName,
    ComponentRegistry,
    ConfigDict,
    ConfigValue,
    EnvMapping,
    EnvVarType,
    FormatStr,
    JsonDict,
    JsonList,
    JsonPrimitive,
    JsonValue,
    LogFilePathStr,
    LoggingConfigDict,
    PathLike,
    SerializedConfig,
    ValidationError,
    ValidationFunction,
)

from .errors import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
)

from .workers import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    ExecutionMetrics,
    TaskPriority,
    TracingContext,
    WorkerState,
    measure_execution,
)

from .protocols import (
    ConfigComponent,
    ConfigComponentInfo,
    JSONSerializable,
    QueueProcessor,
    WorkDistributor,
)

from .templates import (
    # Templates
    InstructionTemplate,
    SQLitePragmas,
    SQLTemplates,
    TemplateDict,
    # Lexical types
    WordnetEntry,
    DictionaryEntry,
    DbnaryEntry,
    LexicalDataset,
    # Graph types
    WordTupleDict,
    RelationshipTupleDict,
    GraphInfoDict,
    # Worker types
    WorkerPoolConfig,
    TaskContext,
    WorkerMode,
    BackpressureStrategy,
    # Domain-specific literal types
    QueryType,
    SQLQueryType,
    EmotionRange,
    SampleWord,
    SampleRelationship,
    LockType,
    QueueMetricsFormat,
    WorkerPoolStrategy,
    OptimizationStrategy,
    BatchingStrategy,
    # Conversation types
    ConversationStatusValue,
    ConversationStatusMap,
    ConversationMetadataSchema,
    # Vector types
    VectorSearchStrategy,
    VectorDistanceMetric,
    VectorOptimizationLevel,
    # Graph export/strategy types
    GraphExportFormat,
    GraphNodeSizeStrategy,
    GraphEdgeWeightStrategy,
    # System types
    TransactionIsolationLevel,
    ConnectionPoolMode,
    LogLevel,
)

from .enums import (
    EnumWithRepr,
    StorageType,
    QueuePerformanceProfile,
    ConversationRetentionPolicy,
    ConversationExportFormat,
    VectorModelType,
    VectorIndexStatus,
    GraphLayoutAlgorithm,
    GraphColorScheme,
    LogFormatTemplate,
    LogRotationStrategy,
    LogDestination,
    DatabaseDialect,
)

from .exceptions import (
    # Config errors
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
    # Worker errors
    WorkerError,
    TaskExecutionError,
    QueueOperationError,
    CircuitOpenError,
    EmptyQueueError,
)

__all__ = [
    # Type variables
    "C", "K", "V", "E", "R", "T", "T_contra",
    # Paths
    "DATA_ROOT", "LOGS_ROOT", "PROJECT_ROOT",
    # Basic types
    "ComponentName", "ComponentRegistry", "ConfigDict", "ConfigValue",
    "EnvMapping", "EnvVarType", "FormatStr", "JsonDict", "JsonList",
    "JsonPrimitive", "JsonValue", "LogFilePathStr", "LoggingConfigDict",
    "PathLike", "SerializedConfig", "ValidationError", "ValidationFunction",
    # Errors
    "Error", "ErrorCategory", "ErrorSeverity", "Result",
    # Workers
    "CircuitBreakerConfig", "CircuitBreakerState", "ExecutionMetrics",
    "TaskPriority", "TracingContext", "WorkerState", "measure_execution",
    # Protocols
    "ConfigComponent", "ConfigComponentInfo", "JSONSerializable",
    "QueueProcessor", "WorkDistributor",
    # Templates
    "InstructionTemplate", "SQLitePragmas", "SQLTemplates", "TemplateDict",
    "WordnetEntry", "DictionaryEntry", "DbnaryEntry", "LexicalDataset",
    "WordTupleDict", "RelationshipTupleDict", "GraphInfoDict",
    "WorkerPoolConfig", "TaskContext", "WorkerMode", "BackpressureStrategy",
    # Domain-specific literal types
    "QueryType", "SQLQueryType", "EmotionRange", "SampleWord", "SampleRelationship",
    "LockType", "QueueMetricsFormat", "WorkerPoolStrategy", "OptimizationStrategy",
    "BatchingStrategy",
    # Conversation types
    "ConversationStatusValue", "ConversationStatusMap", "ConversationMetadataSchema",
    # Vector types
    "VectorSearchStrategy", "VectorDistanceMetric", "VectorOptimizationLevel",
    # Graph types
    "GraphExportFormat", "GraphNodeSizeStrategy", "GraphEdgeWeightStrategy",
    # System types
    "TransactionIsolationLevel", "ConnectionPoolMode", "LogLevel",
    # Enums
    "EnumWithRepr", "StorageType", "QueuePerformanceProfile",
    "ConversationRetentionPolicy", "ConversationExportFormat",
    "VectorModelType", "VectorIndexStatus", "GraphLayoutAlgorithm",
    "GraphColorScheme", "LogFormatTemplate", "LogRotationStrategy",
    "LogDestination", "DatabaseDialect",
    # Exceptions
    "ConfigError", "PathError", "EnvVarError", "VectorConfigError",
    "VectorIndexError", "GraphConfigError", "LoggingConfigError",
    "DatabaseConfigError", "DatabaseConnectionError",
    "LexicalResourceError", "ResourceNotFoundError", "ResourceParsingError",
    "ModelError", "WorkerError", "TaskExecutionError", "QueueOperationError",
    "CircuitOpenError", "EmptyQueueError",
]
