"""
Configuration Essentials Module

This module serves as the type system foundation for WordForge's configuration architecture,
establishing a comprehensive type system that enforces correctness through static analysis
rather than runtime checking.

Key Features:
    - Type-safe configuration definitions with strict interfaces
    - Result pattern for exception-free error propagation
    - Comprehensive serialization utilities for configuration persistence
    - Protocol-based interfaces for consistent component behaviors
    - Fine-grained error classification system with context preservation
    - Thread-safe execution measurement and distributed tracing support

Type Categories:
    - Json types: Primitives, dictionaries, and lists for serialization
    - Configuration types: Component registries and environment mappings
    - Result types: Monadic error handling patterns
    - Worker types: Task priority, worker state management, circuit breakers
    - Domain-specific types: Query types, vector operations, graph processing
    - Template types: Structured definitions for consistent formatting

Performance Support:
    - Execution metrics collection for critical operations
    - Circuit breaker pattern implementation for failure isolation
    - Worker pool configurations for parallel processing
    - Tracing context for distributed operation monitoring

Error Handling:
    - Fine-grained error categorization (validation, resource, business, etc.)
    - Severity classification for appropriate handling strategies
    - Context preservation for diagnostic traceability
    - Monadic Result type for functional error propagation

Usage Guidelines:
    - Import specific types rather than using wildcard imports
    - Leverage type checking tools (mypy, pyright) with this module
    - Follow the Result pattern for cross-component error handling
    - Use the provided serialization utilities for configuration persistence

This module embodies the principle that strong typing prevents more errors than
exception handling ever could, making it the foundation of WordForge's reliability.
This module defines all essential types and constants used throughout the
configuration system, providing a consistent way to handle various
configuration-related tasks, including serialization, path management,
and error handling.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Generic,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

# ==========================================
# Generic Type Variables
# ==========================================

# Generic type parameter for configuration value types
T = TypeVar("T")
# Define a contravariant type variable for input types
T_contra = TypeVar("T_contra", contravariant=True)

# Generic type parameter for function return types
R = TypeVar("R")

# Type variable bound to ConfigComponent protocol for generic configuration handling
C = TypeVar("C", bound="ConfigComponent")

# Additional generic variables for functional patterns
K = TypeVar("K")  # Key type for mappings
V = TypeVar("V")  # Value type for mappings
E = TypeVar("E")  # Error type for Result pattern

# ==========================================
# Project Paths
# ==========================================

# Define project paths relative to this file for portability
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"

# ==========================================
# Basic Type Definitions
# ==========================================

# JSON-related type definitions for configuration serialization
JsonPrimitive = Union[str, int, float, bool, None]
JsonDict = Dict[str, "JsonValue"]  # Forward reference for recursion
JsonList = List["JsonValue"]  # Forward reference for recursion
JsonValue: TypeAlias = Union[JsonDict, JsonList, JsonPrimitive]  # Explicit TypeAlias

# Configuration-specific type aliases
ConfigValue: TypeAlias = JsonValue  # Use defined JsonValue

# Logging
LoggingConfigDict: TypeAlias = Dict[str, Any]  # Define LoggingConfig as Dict
ValidationError: TypeAlias = str
FormatStr: TypeAlias = str
LogFilePathStr: TypeAlias = Optional[str]

# Function type for validation handlers
ValidationFunction: TypeAlias = Callable[
    [LoggingConfigDict, List[ValidationError]], None
]  # Use LoggingConfigDict
EnvVarType: TypeAlias = Union[str, int, float, bool, None]  # Define EnvVarType

# Type alias for serialized configuration data
SerializedConfig: TypeAlias = JsonDict

# Type alias for path-like objects
PathLike: TypeAlias = Union[str, Path]

# Type alias for environment variable mapping in ConfigComponent
EnvMapping: TypeAlias = Dict[str, Tuple[str, EnvVarType]]

# Type alias for the name of a configuration component
ComponentName: TypeAlias = str

# Type alias for a registry mapping component names to instances
ComponentRegistry: TypeAlias = Dict[ComponentName, "ConfigComponent"]

# Type alias for a dictionary representing a configuration section
ConfigDict: TypeAlias = Dict[str, ConfigValue]
# ==========================================
# Result and Error Handling Types
# ==========================================


class ErrorCategory(Enum):
    """Categories of errors for systematic handling and reporting."""

    VALIDATION = auto()  # Input validation failures
    RESOURCE = auto()  # Resource availability issues
    BUSINESS = auto()  # Business rule violations
    EXTERNAL = auto()  # External system failures
    UNEXPECTED = auto()  # Unexpected failures
    CONFIGURATION = auto()  # Configuration errors
    SECURITY = auto()  # Security-related issues


class ErrorSeverity(Enum):
    """Severity levels for errors to guide handling strategies."""

    FATAL = auto()  # System cannot continue operation
    ERROR = auto()  # Operation failed completely
    WARNING = auto()  # Operation completed with issues
    INFO = auto()  # Operation completed with non-critical adjustments


@dataclass(frozen=True)
class Error:
    """
    Immutable error object with comprehensive context for accurate diagnostics.

    Provides a unified structure for error handling across the system,
    with severity classification, error codes, and contextual information.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        category: Error category for classification
        severity: Error severity for handling strategy
        context: Dictionary of additional contextual information
        trace: Optional stack trace for debugging
    """

    message: str
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, str] = field(default_factory=dict)
    trace: Optional[str] = None

    @classmethod
    def create(
        cls,
        message: str,
        code: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[Dict[str, str]] = None,
    ) -> "Error":
        """
        Factory method for creating errors with standardized formatting.

        Args:
            message: Human-readable error description
            code: Machine-readable error code
            category: Error category for classification
            severity: Error severity for handling strategy
            context: Optional dictionary of contextual information

        Returns:
            Fully initialized Error object
        """
        import traceback

        return cls(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context or {},
            trace=traceback.format_exc(),
        )


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    Monadic result type for error handling without exceptions.

    Implements the Result pattern for expressing success/failure
    without raising exceptions across component boundaries.

    Attributes:
        value: The success value (None if error)
        error: The error details (None if success)
    """

    value: Optional[T] = None
    error: Optional[Error] = None

    @property
    def is_success(self) -> bool:
        """Determine if this is a successful result."""
        return self.error is None

    @property
    def is_failure(self) -> bool:
        """Determine if this is a failed result."""
        return not self.is_success

    def unwrap(self) -> T:
        """
        Extract the success value, raising an exception if this is an error result.

        Returns:
            The contained success value

        Raises:
            ValueError: If this is an error result
        """
        if not self.is_success:
            error_msg = f"Cannot unwrap failed result: {self.error.message if self.error else 'Unknown error'}"
            raise ValueError(error_msg)
        return cast(T, self.value)

    def unwrap_or(self, default: T) -> T:
        """
        Extract the success value or return a default if this is an error result.

        Args:
            default: The default value to return if this is an error result

        Returns:
            The contained success value or the provided default
        """
        if not self.is_success:
            return default
        return cast(T, self.value)

    def map(self, f: Callable[[T], R]) -> "Result[R]":
        """
        Apply a function to the value if present, otherwise pass through error.

        Args:
            f: Function to apply to the success value

        Returns:
            A new Result containing either the transformed value or the original error
        """
        if not self.is_success:
            # Type cast needed to preserve type safety with generic error
            return cast(Result[R], Result(error=self.error))
        return Result(value=f(cast(T, self.value)))

    def flat_map(self, f: Callable[[T], "Result[R]"]) -> "Result[R]":
        """
        Apply a function that returns a Result, flattening the result.

        Args:
            f: Function returning a Result to apply to the success value

        Returns:
            The Result returned by the function, or the original error
        """
        if not self.is_success:
            # Type cast needed to preserve type safety with generic error
            return cast(Result[R], Result(error=self.error))
        return f(cast(T, self.value))

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """
        Create a successful result with the given value.

        Args:
            value: The success value

        Returns:
            A successful Result containing the value
        """
        return cls(value=value)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        context: Optional[Dict[str, str]] = None,
        category: ErrorCategory = ErrorCategory.UNEXPECTED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> "Result[T]":
        """
        Create a failure result with the given error details.

        Args:
            code: Machine-readable error code
            message: Human-readable error message
            context: Optional dictionary of contextual information
            category: Error category (default: UNEXPECTED)
            severity: Error severity (default: ERROR)

        Returns:
            A failure Result containing the error details
        """
        error = Error.create(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context,
        )
        return cls(error=error)


# ==========================================
# Task and Worker Types
# ==========================================


class TaskPriority(Enum):
    """Priority levels for scheduling tasks in the work distribution system."""

    CRITICAL = 0  # Must be processed immediately
    HIGH = 1  # Process before normal tasks
    NORMAL = 2  # Default priority
    LOW = 3  # Process after other tasks
    BACKGROUND = 4  # Process only when system is idle


class WorkerState(Enum):
    """States for worker threads in the processing system."""

    INITIALIZING = auto()  # Worker is initializing resources
    IDLE = auto()  # Worker is waiting for tasks
    PROCESSING = auto()  # Worker is processing a task
    PAUSED = auto()  # Worker is temporarily paused
    STOPPING = auto()  # Worker is in the process of stopping
    STOPPED = auto()  # Worker has stopped
    ERROR = auto()  # Worker encountered an error


class CircuitBreakerState(Enum):
    """States for the circuit breaker pattern to prevent cascading failures."""

    CLOSED = auto()  # Normal operation, requests allowed
    OPEN = auto()  # Failing, rejecting requests
    HALF_OPEN = auto()  # Testing if system has recovered


@dataclass
class ExecutionMetrics:
    """
    Metrics collected during operation execution for performance monitoring.

    Provides detailed information about execution time and resource usage
    to help identify bottlenecks and performance issues.

    Attributes:
        operation_name: Name of the operation being measured
        duration_ms: Wall clock execution time in milliseconds
        cpu_time_ms: CPU time used in milliseconds
        memory_delta_kb: Memory usage change in kilobytes
        thread_id: ID of the thread executing the operation
        started_at: Timestamp when execution started
        context: Dictionary of additional context for the operation
    """

    operation_name: str
    duration_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_delta_kb: float = 0.0
    thread_id: int = 0
    started_at: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for a circuit breaker to prevent system overload.

    Controls the behavior of the circuit breaker pattern implementation
    that protects system components from cascading failures.

    Attributes:
        failure_threshold: Number of failures before circuit opens
        reset_timeout_ms: How long to wait before testing circuit again
        half_open_max_calls: Maximum calls allowed in half-open state
        call_timeout_ms: Timeout for individual calls
    """

    failure_threshold: int = 5
    reset_timeout_ms: int = 30000
    half_open_max_calls: int = 3
    call_timeout_ms: int = 5000


class TracingContext:
    """
    Thread-local storage for distributed tracing information.

    Maintains trace and span IDs across async boundaries to enable
    distributed request tracing throughout the system.
    """

    _thread_local = threading.local()

    @classmethod
    def get_current_trace_id(cls) -> Optional[str]:
        """Get the current trace ID from thread-local storage."""
        return getattr(cls._thread_local, "trace_id", None)

    @classmethod
    def get_current_span_id(cls) -> Optional[str]:
        """Get the current span ID from thread-local storage."""
        return getattr(cls._thread_local, "span_id", None)

    @classmethod
    def set_trace_context(cls, trace_id: str, span_id: str) -> None:
        """
        Set the current trace and span IDs.

        Args:
            trace_id: Trace identifier for the current request
            span_id: Span identifier for the current operation
        """
        cls._thread_local.trace_id = trace_id
        cls._thread_local.span_id = span_id

    @classmethod
    def clear_trace_context(cls) -> None:
        """Clear the current trace context."""
        cls._thread_local.trace_id = None
        cls._thread_local.span_id = None


@contextmanager
def measure_execution(
    operation_name: str, context: Optional[Dict[str, Any]] = None
) -> Iterator[ExecutionMetrics]:
    """
    Context manager to measure execution time and resource usage.

    Tracks wall clock time, CPU time, and optionally memory usage for
    performance monitoring of critical operations.

    Args:
        operation_name: Unique identifier for the operation
        context: Optional contextual information

    Yields:
        ExecutionMetrics object that will be populated when context exits
    """
    # Create metrics object
    metrics = ExecutionMetrics(
        operation_name=operation_name,
        thread_id=threading.get_ident(),
        started_at=time.time(),
        context=context or {},
    )

    # Record starting time
    start_time = time.time()
    start_cpu_time = time.process_time()

    try:
        # Yield control back to the caller
        yield metrics
    finally:
        # Calculate durations
        metrics.duration_ms = (time.time() - start_time) * 1000
        metrics.cpu_time_ms = (time.process_time() - start_cpu_time) * 1000


# ==========================================
# Domain-Specific Type Definitions
# ==========================================

# Query and SQL-related types
QueryType = Literal["search", "definition", "similarity"]
SQLQueryType = Literal["get_term_by_id", "get_message_text"]

# Emotion configuration types
EmotionRange = Tuple[float, float]  # (valence, arousal) pairs in range [-1.0, 1.0]

# Sample data types for testing and initialization
SampleWord = Tuple[str, str, str]  # term, definition, part_of_speech
SampleRelationship = Tuple[str, str, str]  # word1, word2, relationship_type

# Queue and concurrency types
LockType = Literal["reentrant", "standard"]
QueueMetricsFormat = Literal["json", "csv", "prometheus"]

# Worker configuration types
WorkerMode = Literal["dedicated", "shared", "adaptive"]
WorkerPoolStrategy = Literal["fixed", "elastic", "workstealing"]
BackpressureStrategy = Literal["reject", "delay", "shed_load", "prioritize"]

# Performance optimization types
OptimizationStrategy = Literal["latency", "throughput", "memory", "balanced"]
BatchingStrategy = Literal["fixed", "dynamic", "adaptive", "none"]

# ==========================================
# Conversation Type Definitions
# ==========================================

# Valid status values for conversations
ConversationStatusValue = Literal[
    "active", "pending", "completed", "archived", "deleted"
]

# Mapping of internal status codes to human-readable descriptions
ConversationStatusMap = Dict[ConversationStatusValue, str]

# Metadata structure for conversation storage
ConversationMetadataSchema = Dict[str, Union[str, int, float, bool, None]]

# ==========================================
# Vector Operations Type Definitions
# ==========================================

# Vector search strategies
VectorSearchStrategy = Literal["exact", "approximate", "hybrid"]

# Vector distance metrics
VectorDistanceMetric = Literal["cosine", "euclidean", "dot", "manhattan"]

# Vector optimization level for tradeoff between speed and accuracy
VectorOptimizationLevel = Literal["speed", "balanced", "accuracy"]

# ==========================================
# Graph Type Definitions
# ==========================================

# Graph export format types
GraphExportFormat = Literal["graphml", "gexf", "json", "png", "svg", "pdf"]

# Graph node size calculation methods
GraphNodeSizeStrategy = Literal["degree", "centrality", "pagerank", "uniform"]

# Edge weight calculation methods
GraphEdgeWeightStrategy = Literal["count", "similarity", "custom"]

# ==========================================
# System Type Definitions
# ==========================================

# Logging level types and utilities
LogLevel = int  # Direct mapping to standard logging module levels
LogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]

# Common log level values for type safety and auto-completion
LOG_LEVEL_DEBUG: Final[LogLevel] = logging.DEBUG
LOG_LEVEL_INFO: Final[LogLevel] = logging.INFO
LOG_LEVEL_WARNING: Final[LogLevel] = logging.WARNING
LOG_LEVEL_ERROR: Final[LogLevel] = logging.ERROR
LOG_LEVEL_CRITICAL: Final[LogLevel] = logging.CRITICAL
LOG_LEVEL_NOTSET: Final[LogLevel] = logging.NOTSET

# Bidirectional mapping between string names and integer values
LOG_LEVEL_MAP: Final[Dict[LogLevelLiteral, LogLevel]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NOTSET": logging.NOTSET,
}


def get_log_level(level: Union[LogLevel, LogLevelLiteral, str]) -> LogLevel:
    """Convert various log level representations to standard logging module integer values.

    Args:
        level: Log level as integer, literal, or string

    Returns:
        Standard integer log level from the logging module

    Raises:
        ValueError: If the provided level string is not recognized
    """
    if isinstance(level, int):
        return level

    upper_level = str(level).upper()
    if upper_level in LOG_LEVEL_MAP:
        return LOG_LEVEL_MAP[upper_level]

    raise ValueError(f"Unknown log level: {level}")


# Database transaction isolation levels
TransactionIsolationLevel = Literal[
    "READ_UNCOMMITTED", "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"
]

# Connection pool modes
ConnectionPoolMode = Literal["fixed", "dynamic", "none"]

# Generic JSON data structure (used for external API responses)
JsonData = Union[Dict[str, JsonValue], List[JsonValue], JsonPrimitive]

# ==========================================
# Configuration Protocols
# ==========================================


class ConfigComponent(Protocol):
    """Protocol defining interface for all configuration components.

    All configuration components must implement this protocol to ensure
    consistency across the system, especially for environment variable
    overriding operations.

    Attributes:
        ENV_VARS: Class variable mapping environment variable names to
                 attribute names and their expected types for overriding
                 configuration values from environment.

    Example:
        ```python
        @dataclass
        class DatabaseConfig:
            db_path: str = "data/wordforge.db"
            pool_size: int = 5

            ENV_VARS: ClassVar[Dict[str, Tuple[str, EnvVarType]]] = {
                "WORDFORGE_DB_PATH": ("db_path", str),
                "WORDFORGE_DB_POOL_SIZE": ("pool_size", int),
            }
        ```
    """

    # Each component must have ENV_VARS class variable for env overrides
    ENV_VARS: ClassVar[Dict[str, Tuple[str, EnvVarType]]]


class JSONSerializable(Protocol):
    """Protocol for objects that can be serialized to JSON.

    Types implementing this protocol can be converted to JSON-compatible
    string representations for storage, transmission, or display purposes.

    Example:
        ```python
        class ConfigObject(JSONSerializable):
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def __str__(self) -> str:
                return f"{{'name': '{self.name}', 'value': {self.value}}}"
        ```
    """

    def __str__(self) -> str:
        """Convert object to string representation for serialization.

        Returns:
            str: A string representation suitable for JSON serialization
        """
        ...


class QueueProcessor(Protocol[T_contra]):
    """
    Protocol defining the interface for queue processing components.

    Implementations define how items from a queue are processed,
    providing a consistent interface for worker threads.
    """

    def process(self, item: T_contra) -> Result[bool]:
        """
        Process an item from the queue.

        Args:
            item: The item to process

        Returns:
            Result indicating success or failure with context
        """
        ...


class WorkDistributor(Protocol):
    """
    Protocol for work distribution systems that manage task parallelism.

    Defines the interface for submitting tasks to a pool of worker threads
    and retrieving results, with support for priorities and backpressure.
    """

    def submit(
        self, task: Any, priority: TaskPriority = TaskPriority.NORMAL
    ) -> Result[Any]:
        """
        Submit a task for processing with the specified priority.

        Args:
            task: The task to process
            priority: The task priority

        Returns:
            Result containing the task result or error
        """
        ...

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """
        Shut down the work distributor.

        Args:
            wait: Whether to wait for pending tasks to complete
            cancel_pending: Whether to cancel pending tasks
        """
        ...


# ==========================================
# Template and Schema Definitions
# ==========================================


class InstructionTemplate(NamedTuple):
    """
    Template structure for model instructions.

    Used to format prompts for embedding models and other generative tasks
    with consistent structure.

    Attributes:
        task: The instruction task description
        query_prefix: Template for prefixing queries
        document_prefix: Optional template for prefixing documents

    Example:
        ```python
        template = InstructionTemplate(
            task="Find documents that answer this question",
            query_prefix="Question: ",
            document_prefix="Document: "
        )
        ```
    """

    task: str
    query_prefix: str
    document_prefix: Optional[str] = None


class SQLitePragmas(TypedDict, total=False):
    """
    Type definition for SQLite pragma settings with precise performance control.

    Provides a comprehensive, type-safe structure for SQLite behavior configuration,
    allowing fine-grained control over database performance characteristics,
    concurrency behavior, and data integrity guarantees.

    Each pragma represents a specific optimization dimension with carefully constrained
    valid values. This structure prevents configuration errors through type checking
    rather than runtime exceptions—embodying the Eidosian principle that architecture
    prevents errors better than exception handling.

    Attributes:
        foreign_keys: Enable/disable foreign key constraints ("ON"/"OFF")
        journal_mode: Transaction journaling mode ("WAL", "DELETE", "MEMORY", "OFF", "PERSIST", "TRUNCATE")
        synchronous: Disk synchronization strategy ("NORMAL", "FULL", "OFF", "EXTRA")
        cache_size: Database cache size in pages or KiB (positive for pages, negative for KiB)
        temp_store: Temporary storage location ("MEMORY", "FILE", "DEFAULT")
        mmap_size: Memory map size in bytes for file access optimization (e.g., "1073741824" for 1GB)
    """

    foreign_keys: str
    journal_mode: str
    synchronous: str
    cache_size: str
    temp_store: str
    mmap_size: str


class SQLTemplates(TypedDict, total=False):
    """SQL query templates for graph operations."""

    check_words_table: str
    check_relationships_table: str
    fetch_all_words: str
    fetch_words_since: str
    fetch_all_relationships: str
    fetch_relationships_since: str
    get_all_words: str
    get_all_relationships: str
    get_emotional_relationships: str
    get_all_emotional_relationships: str
    get_emotional_relationships_since: str
    insert_sample_word: str
    insert_sample_relationship: str


class TemplateDict(TypedDict):
    """
    Structure defining an instruction template configuration.

    Used for configuring instruction templates through configuration files
    rather than direct instantiation.

    Attributes:
        task: The instruction task description
        query_prefix: Template for prefixing queries
        document_prefix: Optional template for prefixing documents
    """

    task: Optional[str]
    query_prefix: Optional[str]
    document_prefix: Optional[str]


class WordnetEntry(TypedDict):
    """
    Type definition for a WordNet entry with comprehensive lexical information.

    Structured representation of WordNet data used in the parser and database.

    Attributes:
        word: The lexical item itself
        definition: Word definition
        examples: Usage examples for this word
        synonyms: List of synonym words
        antonyms: List of antonym words
        part_of_speech: Grammatical category (noun, verb, etc.)
    """

    word: str
    definition: str
    examples: List[str]
    synonyms: List[str]
    antonyms: List[str]
    part_of_speech: str


class DictionaryEntry(TypedDict):
    """
    Type definition for a standard dictionary entry.

    Generic dictionary format used for various data sources.

    Attributes:
        definition: The word definition
        examples: Usage examples for this word
    """

    definition: str
    examples: List[str]


class DbnaryEntry(TypedDict):
    """
    Type definition for a DBnary lexical entry containing definitions and translations.

    Specialized structure for multilingual dictionary entries.

    Attributes:
        definition: Word definition
        translation: Translation in target language
    """

    definition: str
    translation: str


class LexicalDataset(TypedDict):
    """
    Type definition for the comprehensive lexical dataset.

    Consolidated data structure containing information from multiple sources
    for a single lexical item.

    Attributes:
        word: The lexical item itself
        wordnet_data: Data from WordNet
        openthesaurus_synonyms: List[str]
        odict_data: DictionaryEntry
        dbnary_data: List[DbnaryEntry]
        opendict_data: DictionaryEntry
        thesaurus_synonyms: List[str]
        example_sentence: Example usage in context
    """

    word: str
    wordnet_data: List[WordnetEntry]
    openthesaurus_synonyms: List[str]
    odict_data: DictionaryEntry
    dbnary_data: List[DbnaryEntry]
    opendict_data: DictionaryEntry
    thesaurus_synonyms: List[str]
    example_sentence: str


class WordTupleDict(TypedDict):
    """Dictionary representation of a word node in the graph.

    Args:
        id: Unique identifier for the word
        term: The actual word or lexical item
        pos: Part of speech tag (optional)
        frequency: Word usage frequency (optional)
    """

    id: int
    term: str
    pos: Optional[str]
    frequency: Optional[float]


class RelationshipTupleDict(TypedDict):
    """Dictionary representation of a relationship between words.

    Args:
        source_id: ID of the source word
        target_id: ID of the target word
        rel_type: Type of relationship (e.g., "synonym", "antonym")
        weight: Strength of the relationship (0.0 to 1.0)
        dimension: Semantic dimension of the relationship
        bidirectional: Whether relationship applies in both directions
    """

    source_id: int
    target_id: int
    rel_type: str
    weight: float
    dimension: str
    bidirectional: bool


class GraphInfoDict(TypedDict):
    """Dictionary containing graph metadata and statistics.

    Args:
        node_count: Total number of nodes in the graph
        edge_count: Total number of edges in the graph
        density: Graph density measurement
        dimensions: Set of relationship dimensions present
        rel_types: Dictionary mapping relationship types to counts
        connected_components: Number of connected components
        largest_component_size: Size of the largest connected component
    """

    node_count: int
    edge_count: int
    density: float
    dimensions: Set[str]
    rel_types: Dict[str, int]
    connected_components: int
    largest_component_size: int


class WorkerPoolConfig(TypedDict):
    """
    Configuration for a worker thread pool.

    Controls the behavior of the parallel processing system, including
    number of workers, queue size, and backpressure strategy.

    Attributes:
        worker_count: Number of worker threads
        max_queue_size: Maximum number of items in the work queue
        worker_mode: Worker thread allocation strategy
        batch_size: Number of items to process in a batch
        backpressure_strategy: Strategy for handling queue overflow
    """

    worker_count: int
    max_queue_size: int
    worker_mode: WorkerMode
    batch_size: int
    backpressure_strategy: BackpressureStrategy


class TaskContext(TypedDict, total=False):
    """
    Context information for task execution.

    Provides additional information to worker threads about how to
    process a task, including tracing data and execution constraints.

    Attributes:
        trace_id: Distributed tracing identifier
        timeout_ms: Maximum execution time in milliseconds
        retry_count: Number of times this task has been retried
        service_name: Name of the service processing this task
        created_at: Timestamp when the task was created
    """

    trace_id: str
    timeout_ms: int
    retry_count: int
    service_name: str
    created_at: float


# ==========================================
# Configuration Component Metadata
# ==========================================


@dataclass(frozen=True)
class ConfigComponentInfo:
    """
    Metadata about a configuration component.

    Used to track component relationships and dependencies for reflection,
    dependency resolution, and runtime validation.

    Attributes:
        name: Component name used for registry lookup
        class_type: The class of the component for type checking
        dependencies: Names of other components this one depends on

    Example:
        ```python
        info = ConfigComponentInfo(
            name="database",
            class_type=DatabaseConfig,
            dependencies={"logging"}
        )
        ```
    """

    name: str
    class_type: Type[ConfigComponent]
    dependencies: Set[str] = field(default_factory=set)


# ==========================================
# Enum Definitions
# ==========================================


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

    LOW_LATENCY = "low_latency"  # Optimize for immediate response
    HIGH_THROUGHPUT = "high_throughput"  # Optimize for maximum processing volume
    BALANCED = "balanced"  # Balance between latency and throughput
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage


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

    TRANSFORMER = "transformer"  # Transformer-based embedding models
    SENTENCE = "sentence"  # Sentence embedding models
    WORD = "word"  # Word embedding models
    CUSTOM = "custom"  # Custom embedding implementations


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

    UNINITIALIZED = "uninitialized"  # Index has not been created
    READY = "ready"  # Index is ready for use
    BUILDING = "building"  # Index is currently being built
    ERROR = "error"  # Index encountered an error


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

    FORCE_DIRECTED = "force_directed"  # Force-directed graph drawing
    CIRCULAR = "circular"  # Circular layout
    HIERARCHICAL = "hierarchical"  # Tree-like hierarchical layout
    SPECTRAL = "spectral"  # Spectral layout using eigenvectors
    RADIAL = "radial"  # Radial layout around central node


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

    SEMANTIC = "semantic"  # Colors based on semantic relationships
    CATEGORY = "category"  # Colors based on word categories
    SENTIMENT = "sentiment"  # Colors based on sentiment analysis
    GRADIENT = "gradient"  # Gradient colors based on relationship strength
    MONOCHROME = "monochrome"  # Single color with varying intensity


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

    SIZE = "size"  # Rotate based on file size
    TIME = "time"  # Rotate based on time intervals
    NONE = "none"  # No rotation


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

    CONSOLE = "console"  # Log to console only
    FILE = "file"  # Log to file only
    BOTH = "both"  # Log to both console and file
    SYSLOG = "syslog"  # Log to system log


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

    SQLITE = "sqlite"  # SQLite file-based database
    POSTGRES = "postgres"  # PostgreSQL database
    MYSQL = "mysql"  # MySQL database
    MEMORY = "memory"  # In-memory database (for testing)


# ==========================================
# Exception Definitions
# ==========================================


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class PathError(ConfigError):
    """Raised when a path operation fails."""

    pass


class EnvVarError(ConfigError):
    """Raised when an environment variable cannot be processed."""

    pass


class VectorConfigError(ConfigError):
    """Raised when vector configuration is invalid."""

    pass


class VectorIndexError(ConfigError):
    """Raised when vector index operations fail."""

    pass


class GraphConfigError(ConfigError):
    """Raised when graph configuration is invalid."""

    pass


class LoggingConfigError(ConfigError):
    """Raised when logging configuration is invalid."""

    pass


class DatabaseConfigError(ConfigError):
    """Raised when database configuration is invalid."""

    pass


class DatabaseConnectionError(ConfigError):
    """Raised when database connection fails."""

    pass


class LexicalResourceError(Exception):
    """Exception raised when a lexical resource cannot be accessed or processed."""

    pass


class ResourceNotFoundError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be found."""

    pass


class ResourceParsingError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be parsed."""

    pass


class ModelError(Exception):
    """Exception raised when there's an issue with the language model."""

    pass


class WorkerError(Exception):
    """Base exception for worker thread errors."""

    pass


class TaskExecutionError(WorkerError):
    """Raised when a task fails during execution."""

    pass


class QueueOperationError(WorkerError):
    """Raised when a queue operation fails."""

    pass


class CircuitOpenError(WorkerError):
    """Raised when an operation is rejected due to an open circuit breaker."""

    pass


class EmptyQueueError(QueueOperationError):
    """Raised when attempting to dequeue from an empty queue."""

    pass


# ==========================================
# Serialization Utilities
# ==========================================


def serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """
    Serialize a dataclass to a dictionary, handling special types like Enums.

    This function takes a dataclass instance and converts it to a dictionary
    representation suitable for JSON serialization. It handles special cases
    like Enum values and named tuples.

    Args:
        obj: Dataclass instance to serialize

    Returns:
        Dictionary with serialized values where Enums are converted to their values
        and NamedTuples are converted to dictionaries

    Examples:
        >>> from dataclasses import dataclass
        >>> from enum import Enum
        >>>
        >>> class Color(Enum):
        ...     RED = "red"
        ...     BLUE = "blue"
        ...
        >>> @dataclass
        ... class Settings:
        ...     name: str
        ...     color: Color
        ...
        >>> settings = Settings(name="test", color=Color.RED)
        >>> serialize_dataclass(settings)
        {'name': 'test', 'color': 'red'}
    """
    result: Dict[str, Any] = {}
    for key, value in asdict(obj).items():
        if isinstance(value, Enum):
            # Serialize enum as its value
            result[key] = value.value
        elif isinstance(value, tuple):
            # Handle named tuples
            try:
                # Cast to Any to safely check for NamedTuple attributes
                tuple_value = cast(Any, value)
                if hasattr(tuple_value, "_fields") and callable(
                    getattr(tuple_value, "_asdict", None)
                ):
                    result[key] = tuple_value._asdict()
                else:
                    result[key] = value
            except (AttributeError, TypeError):
                result[key] = value
        else:
            result[key] = value
    return result


def serialize_config(obj: Any) -> ConfigValue:
    """
    Convert configuration objects to dictionaries for display or serialization.

    Recursively processes configuration objects for JSON serialization,
    handling special types like Enums, dataclasses, lists, tuples, and dictionaries.

    Args:
        obj: Any configuration object or value to serialize

    Returns:
        A JSON-serializable representation of the configuration

    Examples:
        >>> class Config:
        ...     def __init__(self):
        ...         self.name = "test"
        ...         self.values = [1, 2, 3]
        ...         self._private = "hidden"
        ...
        >>> serialize_config(Config())
        {'name': 'test', 'values': [1, 2, 3]}
    """
    if hasattr(obj, "__dict__"):
        # Use ConfigDict for the dictionary type hint
        d: ConfigDict = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                d[key] = serialize_config(value)
        return d
    elif isinstance(obj, (list, tuple)):
        # Return JsonList type
        return cast(
            JsonList, [serialize_config(item) for item in cast(Sequence[Any], obj)]
        )
    elif isinstance(obj, dict):
        # Return JsonDict type
        return {
            str(key): serialize_config(value)
            for key, value in cast(Dict[Any, Any], obj).items()
        }
    elif isinstance(obj, Enum):
        # Return the enum's value, which should be a JsonPrimitive
        return cast(JsonPrimitive, obj.value)
    elif isinstance(obj, Path):
        # Return string representation of Path
        return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        # Return JsonPrimitive directly
        return cast(JsonPrimitive, obj)
    # Fallback: convert other types to string
    return str(obj)


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Project Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # Type Variables
    "T",
    "R",
    "K",
    "V",
    "E",
    # Configuration Components
    "ConfigComponentInfo",
    # Configurations
    "WorkerPoolConfig",
    # Protocols
    "JSONSerializable",
    "QueueProcessor",
    "WorkDistributor",
    # Serialization Utilities
    "serialize_dataclass",
    "serialize_config",
    # Basic Types
    "JsonPrimitive",
    "JsonDict",
    "JsonList",
    "JsonValue",
    "ConfigValue",
    "SerializedConfig",
    "PathLike",
    "EnvVarType",
    "EnvMapping",
    "ComponentName",
    "ComponentRegistry",
    "ConfigDict",
    "JsonData",
    # Domain-Specific Types
    "QueryType",
    "SQLQueryType",
    "EmotionRange",
    "SampleWord",
    "SampleRelationship",
    "LockType",
    "QueueMetricsFormat",
    "WorkerMode",
    "WorkerPoolStrategy",
    "BackpressureStrategy",
    "OptimizationStrategy",
    "BatchingStrategy",
    # Conversation Types
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    # Vector Operation Types
    "VectorSearchStrategy",
    "VectorDistanceMetric",
    "VectorOptimizationLevel",
    # Graph Types
    "GraphExportFormat",
    "GraphNodeSizeStrategy",
    "GraphEdgeWeightStrategy",
    # System Types
    "LogLevel",
    "TransactionIsolationLevel",
    "ConnectionPoolMode",
    # Error Handling Types
    "Error",
    "ErrorCategory",
    "ErrorSeverity",
    "Result",
    # Worker Types
    "TaskPriority",
    "WorkerState",
    "CircuitBreakerState",
    "ExecutionMetrics",
    "CircuitBreakerConfig",
    "TracingContext",
    "WorkerPoolConfig",
    "TaskContext",
    "measure_execution",
    # Templates and Schemas
    "InstructionTemplate",
    "SQLitePragmas",
    "SQLTemplates",
    "TemplateDict",
    "WordnetEntry",
    "DictionaryEntry",
    "DbnaryEntry",
    "LexicalDataset",
    "WordTupleDict",
    "RelationshipTupleDict",
    "GraphInfoDict",
    # Enum Types
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
    # Error Types
    "ConfigError",
    "PathError",
    "EnvVarError",
    "VectorConfigError",
    "VectorIndexError",
    "GraphConfigError",
    "LoggingConfigError",
    "DatabaseConfigError",
    "DatabaseConnectionError",
    "LexicalResourceError",
    "ResourceNotFoundError",
    "ResourceParsingError",
    "ModelError",
    "WorkerError",
    "TaskExecutionError",
    "QueueOperationError",
    "CircuitOpenError",
    "EmptyQueueError",
    "TypeAlias",
]
