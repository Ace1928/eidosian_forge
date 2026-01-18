"""
Unified type system for Word Forge configuration.

This module defines all types, aliases, and dataclasses used throughout the
configuration system. These types provide strict contracts for configuration
components, ensuring type safety across the entire application.

Type definitions are organized by domain and include comprehensive documentation
to facilitate correct usage throughout the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

# ==========================================
# Project Paths
# ==========================================

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"

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
    class_type: Type[Any]  # Using Any to allow various config component types
    dependencies: Set[str] = field(default_factory=set)


# ==========================================
# Basic Type Definitions
# ==========================================

# Generic type parameters with descriptive names
T = TypeVar("T")  # Generic type for configuration values
R = TypeVar("R")  # Return type for type conversion functions

# JSON-related type definitions for configuration serialization
JsonPrimitive = Union[str, int, float, bool, None]
JsonDict = Dict[str, Any]  # Using Any here due to recursive nature
JsonList = List[Any]  # Using Any here due to recursive nature
JsonValue = Union[JsonDict, JsonList, JsonPrimitive]

# Configuration-specific type aliases
ConfigValue = JsonValue  # Alias for clarity in configuration context
SerializedConfig = Dict[str, ConfigValue]
PathLike = Union[str, Path]

# Environment variable related types
EnvVarType = Union[Type[str], Type[int], Type[float], Type[bool], Type[Enum]]
EnvMapping = Dict[str, Tuple[str, EnvVarType]]

# Component registry related types
ComponentName = str
ComponentRegistry = Dict[ComponentName, ConfigComponentInfo]

# Self-documenting type alias for configuration dictionaries
ConfigDict = Dict[str, Any]

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

# Log levels with strict typing
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Database transaction isolation levels
TransactionIsolationLevel = Literal[
    "READ_UNCOMMITTED", "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"
]

# Connection pool modes
ConnectionPoolMode = Literal["fixed", "dynamic", "none"]

# Generic JSON data structure (used for external API responses)
JsonData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

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
    Type definition for SQLite pragma settings.

    Defines the type-safe structure for SQLite performance and behavior
    configuration options.

    Attributes:
        foreign_keys: Enable/disable foreign key constraints ("ON"/"OFF")
        journal_mode: Transaction journaling mode (WAL, DELETE, etc.)
        synchronous: Disk synchronization strategy (NORMAL, FULL, OFF)
        cache_size: Database cache size in pages or KiB
        temp_store: Temporary storage location (MEMORY, FILE)
    """

    foreign_keys: str
    journal_mode: str
    synchronous: str
    cache_size: str
    temp_store: str


class SQLTemplates(TypedDict):
    """SQL query templates for graph operations."""

    check_words_table: str
    check_relationships_table: str
    fetch_all_words: str
    fetch_all_relationships: str
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
        openthesaurus_synonyms: Synonyms from OpenThesaurus
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


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Project Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # Configuration Components
    "ConfigComponentInfo",
    # Basic Types
    "T",
    "R",
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
    # Domain-Specific Types
    "QueryType",
    "SQLQueryType",
    "EmotionRange",
    "SampleWord",
    "SampleRelationship",
    "LockType",
    "QueueMetricsFormat",
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
]
