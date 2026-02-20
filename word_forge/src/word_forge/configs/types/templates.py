"""
Template and schema type definitions for Word Forge configuration system.

This module provides TypedDict and NamedTuple definitions for structured
data used throughout the configuration system.

Classes:
    InstructionTemplate: Template structure for model instructions
    SQLitePragmas: SQLite pragma settings
    SQLTemplates: SQL query templates
    TemplateDict: Instruction template configuration
    WordnetEntry: WordNet lexical entry
    DictionaryEntry: Standard dictionary entry
    DbnaryEntry: DBnary multilingual entry
    LexicalDataset: Comprehensive lexical dataset
    WordTupleDict: Word node representation
    RelationshipTupleDict: Relationship representation
    GraphInfoDict: Graph metadata and statistics
    WorkerPoolConfig: Worker thread pool configuration
    TaskContext: Task execution context
"""

from typing import (
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)


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


# Worker mode literals
WorkerMode = Literal["thread", "process", "async"]
BackpressureStrategy = Literal["drop", "block", "reject"]


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
# Domain-Specific Literal Types
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

# Worker configuration types (Note: WorkerMode is also defined as TypedDict key above)
WorkerPoolStrategy = Literal["fixed", "elastic", "workstealing"]
OptimizationStrategy = Literal["latency", "throughput", "memory", "balanced"]
BatchingStrategy = Literal["fixed", "dynamic", "adaptive", "none"]

# ==========================================
# Conversation Type Definitions
# ==========================================

# Valid status values for conversations
ConversationStatusValue = Literal["active", "pending", "completed", "archived", "deleted"]

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

# Database transaction isolation levels
TransactionIsolationLevel = Literal["READ_UNCOMMITTED", "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"]

# Connection pool modes
ConnectionPoolMode = Literal["fixed", "dynamic", "none"]

# Logging level types
LogLevel = int  # Direct mapping to standard logging module levels


__all__ = [
    # Templates
    "InstructionTemplate",
    "SQLitePragmas",
    "SQLTemplates",
    "TemplateDict",
    # Lexical types
    "WordnetEntry",
    "DictionaryEntry",
    "DbnaryEntry",
    "LexicalDataset",
    # Graph types
    "WordTupleDict",
    "RelationshipTupleDict",
    "GraphInfoDict",
    # Worker types
    "WorkerPoolConfig",
    "TaskContext",
    "WorkerMode",
    "BackpressureStrategy",
    # Domain-specific literal types
    "QueryType",
    "SQLQueryType",
    "EmotionRange",
    "SampleWord",
    "SampleRelationship",
    "LockType",
    "QueueMetricsFormat",
    "WorkerPoolStrategy",
    "OptimizationStrategy",
    "BatchingStrategy",
    # Conversation types
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    # Vector types
    "VectorSearchStrategy",
    "VectorDistanceMetric",
    "VectorOptimizationLevel",
    # Graph types
    "GraphExportFormat",
    "GraphNodeSizeStrategy",
    "GraphEdgeWeightStrategy",
    # System types
    "TransactionIsolationLevel",
    "ConnectionPoolMode",
    "LogLevel",
]
