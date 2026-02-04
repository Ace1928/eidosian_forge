"""
Enumeration types for Word Forge configuration system.

This module provides all enumeration types used for configuration,
ensuring type-safe configuration values with standardized representations.

Classes:
    EnumWithRepr: Base enum class with standardized string representation
    StorageType: Storage strategy for vector embeddings
    QueuePerformanceProfile: Performance profiles for queue processing
    ConversationRetentionPolicy: Retention policy for conversation history
    ConversationExportFormat: Export format options for conversation data
    VectorModelType: Vector embedding model types
    VectorIndexStatus: Status of a vector index
    GraphLayoutAlgorithm: Layout algorithms for graph visualization
    GraphColorScheme: Color schemes for graph visualization
    LogFormatTemplate: Standard logging format templates
    LogRotationStrategy: Log file rotation strategies
    LogDestination: Logging output destinations
    DatabaseDialect: Database dialects supported
"""

from enum import Enum, auto


class EnumWithRepr(Enum):
    """Base enum class with standardized string representation.

    All enumeration types in the configuration system inherit from this class
    to ensure consistent string representation for debugging and serialization.
    """

    def __repr__(self) -> str:
        """Provide a clean representation for debugging.

        Returns:
            str: String in the format 'EnumClassName.MEMBER_NAME'

        Example:
            >>> repr(StorageType.MEMORY)
            'StorageType.MEMORY'
        """
        return f"{self.__class__.__name__}.{self.name}"


class StorageType(EnumWithRepr):
    """Storage strategy for vector embeddings.

    Defines how vector embeddings are stored and accessed within the system,
    balancing between speed and persistence requirements.

    Attributes:
        MEMORY: In-memory storage for fast access but no persistence
        DISK: Persistent disk-based storage
    """

    MEMORY = "memory"
    DISK = "disk"


class QueuePerformanceProfile(EnumWithRepr):
    """Performance profiles for queue processing operations.

    Defines different optimization strategies for the queue system based
    on the specific performance requirements of the application.

    Attributes:
        LOW_LATENCY: Optimize for immediate response time
        HIGH_THROUGHPUT: Optimize for maximum processing volume
        BALANCED: Balance between latency and throughput
        MEMORY_EFFICIENT: Minimize memory usage
    """

    LOW_LATENCY = "low_latency"
    HIGH_THROUGHPUT = "high_throughput"
    BALANCED = "balanced"
    MEMORY_EFFICIENT = "memory_efficient"


class ConversationRetentionPolicy(EnumWithRepr):
    """Retention policy options for conversation history.

    Defines how long conversation data should be retained in the system
    before automatic deletion occurs.

    Attributes:
        KEEP_FOREVER: Never automatically delete conversation data
        DELETE_AFTER_30_DAYS: Automatically delete after 30 days
        DELETE_AFTER_90_DAYS: Automatically delete after 90 days
        DELETE_AFTER_1_YEAR: Automatically delete after 1 year
    """

    KEEP_FOREVER = "keep_forever"
    DELETE_AFTER_30_DAYS = "delete_after_30_days"
    DELETE_AFTER_90_DAYS = "delete_after_90_days"
    DELETE_AFTER_1_YEAR = "delete_after_1_year"


class ConversationExportFormat(EnumWithRepr):
    """Export format options for conversation data.

    Defines the supported file formats when exporting conversation history
    for external use, archiving, or visualization.

    Attributes:
        JSON: Export as structured JSON data
        MARKDOWN: Export as Markdown formatted text
        TEXT: Export as plain text
        HTML: Export as formatted HTML document
    """

    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"


class VectorModelType(EnumWithRepr):
    """Vector embedding model types supported by the system.

    Categorizes the different approaches to generating vector embeddings
    based on their underlying techniques and capabilities.

    Attributes:
        TRANSFORMER: Transformer-based embedding models (e.g., BERT)
        SENTENCE: Models optimized for sentence-level semantics
        WORD: Word embedding models (e.g., Word2Vec, GloVe)
        CUSTOM: Custom embedding implementations
    """

    TRANSFORMER = "transformer"
    SENTENCE = "sentence"
    WORD = "word"
    CUSTOM = "custom"


class VectorIndexStatus(EnumWithRepr):
    """Status of a vector index.

    Tracks the current state of a vector index throughout its lifecycle,
    from initialization through building to ready state or error condition.

    Attributes:
        UNINITIALIZED: Index has not been created yet
        READY: Index is built and ready for use
        BUILDING: Index is currently being built
        ERROR: Index encountered an error
    """

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    BUILDING = "building"
    ERROR = "error"


class GraphLayoutAlgorithm(EnumWithRepr):
    """Layout algorithms for knowledge graph visualization.

    Defines different algorithms for arranging nodes and edges in
    a knowledge graph visualization to emphasize different structural
    aspects of the graph.

    Attributes:
        FORCE_DIRECTED: Physics-based simulation for natural layouts
        CIRCULAR: Arranges nodes in a circle pattern
        HIERARCHICAL: Tree-like layout for hierarchical data
        SPECTRAL: Layout using graph eigenvectors for clustering
        RADIAL: Arranges nodes around a central node
    """

    FORCE_DIRECTED = "force_directed"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    RADIAL = "radial"


class GraphColorScheme(EnumWithRepr):
    """Color schemes for graph visualization.

    Defines different approaches to coloring nodes and edges in
    a knowledge graph visualization based on various properties
    of the graph elements.

    Attributes:
        SEMANTIC: Colors based on semantic relationship types
        CATEGORY: Colors based on word categories or classifications
        SENTIMENT: Colors based on sentiment analysis values
        GRADIENT: Gradient colors based on relationship strength
        MONOCHROME: Single color with varying intensity levels
    """

    SEMANTIC = "semantic"
    CATEGORY = "category"
    SENTIMENT = "sentiment"
    GRADIENT = "gradient"
    MONOCHROME = "monochrome"


class LogFormatTemplate(EnumWithRepr):
    """Standard logging format templates.

    Predefined formatting strings for log messages that control
    what information is included in each log entry.

    Attributes:
        SIMPLE: Basic format with just the message
        STANDARD: Common format with timestamp, name, level, and message
        DETAILED: Extended format with file and line information
        JSON: Structured JSON format for machine processing
    """

    SIMPLE = "%(message)s"
    STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED = (
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    JSON = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'


class LogRotationStrategy(EnumWithRepr):
    """Log file rotation strategies.

    Defines when log files should be rotated to manage file size
    and organize logging history.

    Attributes:
        SIZE: Rotate based on file size reaching a threshold
        TIME: Rotate based on time intervals (e.g., daily)
        NONE: No rotation, use a single continuous log file
    """

    SIZE = "size"
    TIME = "time"
    NONE = "none"


class LogDestination(EnumWithRepr):
    """Logging output destinations.

    Defines where log messages should be sent for storage
    or display.

    Attributes:
        CONSOLE: Log to standard output/console only
        FILE: Log to file only
        BOTH: Log to both console and file
        SYSLOG: Log to system log facility
    """

    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
    SYSLOG = "syslog"


class DatabaseDialect(EnumWithRepr):
    """Database dialects supported by the system.

    Defines the different database systems that can be used
    as storage backends for the application.

    Attributes:
        SQLITE: SQLite file-based database
        POSTGRES: PostgreSQL database
        MYSQL: MySQL database
        MEMORY: In-memory database (primarily for testing)
    """

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MEMORY = "memory"


__all__ = [
    "EnumWithRepr",
    "StorageType",
    "QueuePerformanceProfile",
    "ConversationRetentionPolicy",
    "ConversationExportFormat",
    "VectorModelType",
    "VectorIndexStatus",
    "GraphLayoutAlgorithm",
    "GraphColorScheme",
    "LogFormatTemplate",
    "LogRotationStrategy",
    "LogDestination",
    "DatabaseDialect",
]
