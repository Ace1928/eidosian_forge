"""
Exception types for Word Forge configuration system.

This module provides all custom exception types used for error handling
throughout the configuration and processing system.

Exception Hierarchy:
    ConfigError (base)
        PathError
        EnvVarError
        VectorConfigError
        VectorIndexError
        GraphConfigError
        LoggingConfigError
        DatabaseConfigError
        DatabaseConnectionError
    
    LexicalResourceError (base)
        ResourceNotFoundError
        ResourceParsingError
    
    ModelError (standalone)
    
    WorkerError (base)
        TaskExecutionError
        QueueOperationError
            EmptyQueueError
        CircuitOpenError
"""


class ConfigError(Exception):
    """Base exception for configuration errors.
    
    All configuration-related exceptions inherit from this class,
    allowing for targeted exception handling of config issues.
    """

    pass


class PathError(ConfigError):
    """Raised when a path operation fails.
    
    This includes issues with file/directory creation, access,
    or path resolution during configuration processing.
    """

    pass


class EnvVarError(ConfigError):
    """Raised when an environment variable cannot be processed.
    
    This includes missing required environment variables, type
    conversion failures, or invalid values.
    """

    pass


class VectorConfigError(ConfigError):
    """Raised when vector configuration is invalid.
    
    This includes issues with embedding dimensions, model selection,
    or storage configuration for vector operations.
    """

    pass


class VectorIndexError(ConfigError):
    """Raised when vector index operations fail.
    
    This includes index creation, building, searching, or
    persistence failures.
    """

    pass


class GraphConfigError(ConfigError):
    """Raised when graph configuration is invalid.
    
    This includes issues with graph storage, visualization settings,
    or relationship configuration.
    """

    pass


class LoggingConfigError(ConfigError):
    """Raised when logging configuration is invalid.
    
    This includes issues with log file paths, rotation settings,
    or format configuration.
    """

    pass


class DatabaseConfigError(ConfigError):
    """Raised when database configuration is invalid.
    
    This includes issues with connection strings, pool settings,
    or pragma configuration.
    """

    pass


class DatabaseConnectionError(ConfigError):
    """Raised when database connection fails.
    
    This is separate from DatabaseConfigError as it indicates
    a runtime failure rather than a configuration issue.
    """

    pass


class LexicalResourceError(Exception):
    """Exception raised when a lexical resource cannot be accessed or processed.
    
    Base class for all lexical resource-related exceptions.
    """

    pass


class ResourceNotFoundError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be found.
    
    This includes missing WordNet data, thesaurus files, or
    other lexical data sources.
    """

    pass


class ResourceParsingError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be parsed.
    
    This includes malformed data files, encoding issues, or
    schema violations in lexical resources.
    """

    pass


class ModelError(Exception):
    """Exception raised when there's an issue with the language model.
    
    This includes model loading failures, inference errors, or
    resource constraints preventing model operations.
    """

    pass


class WorkerError(Exception):
    """Base exception for worker thread errors.
    
    All worker/queue-related exceptions inherit from this class.
    """

    pass


class TaskExecutionError(WorkerError):
    """Raised when a task fails during execution.
    
    This includes exceptions thrown by task handlers, timeout
    violations, or resource exhaustion during task processing.
    """

    pass


class QueueOperationError(WorkerError):
    """Raised when a queue operation fails.
    
    This includes enqueue/dequeue failures, capacity violations,
    or queue corruption issues.
    """

    pass


class CircuitOpenError(WorkerError):
    """Raised when an operation is rejected due to an open circuit breaker.
    
    This indicates the system is in a protective state due to
    repeated failures and new operations are being rejected.
    """

    pass


class EmptyQueueError(QueueOperationError):
    """Raised when attempting to dequeue from an empty queue.
    
    This is a specific case of QueueOperationError for empty
    queue conditions.
    """

    pass


class LexicalResourceError(Exception):
    """Exception raised when a lexical resource cannot be accessed or processed.
    
    Base class for all lexical resource-related exceptions.
    """

    pass


class ResourceNotFoundError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be found.
    
    This includes missing WordNet data, thesaurus files, or
    other lexical data sources.
    """

    pass


class ResourceParsingError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be parsed.
    
    This includes malformed data files, encoding issues, or
    schema violations in lexical resources.
    """

    pass


class ModelError(Exception):
    """Exception raised when there's an issue with the language model.
    
    This includes model loading failures, inference errors, or
    resource constraints preventing model operations.
    """

    pass


__all__ = [
    # Config errors
    "ConfigError",
    "PathError",
    "EnvVarError",
    "VectorConfigError",
    "VectorIndexError",
    "GraphConfigError",
    "LoggingConfigError",
    "DatabaseConfigError",
    "DatabaseConnectionError",
    # Lexical resource errors
    "LexicalResourceError",
    "ResourceNotFoundError",
    "ResourceParsingError",
    # Model errors
    "ModelError",
    # Worker errors
    "WorkerError",
    "TaskExecutionError",
    "QueueOperationError",
    "CircuitOpenError",
    "EmptyQueueError",
]
