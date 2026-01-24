"""
Common exceptions shared across Word Forge modules.

This module provides base exceptions used throughout the Word Forge system,
ensuring consistent error handling and avoiding circular imports.

Error Code Format:
    Each exception has an associated error code following the pattern:
    WF-{CATEGORY}-{NUMBER}

    Categories:
    - DB: Database errors (WF-DB-xxx)
    - GR: Graph errors (WF-GR-xxx)
    - VE: Vector errors (WF-VE-xxx)
    - QU: Queue errors (WF-QU-xxx)
    - PA: Parser errors (WF-PA-xxx)
    - CV: Conversation errors (WF-CV-xxx)
    - GN: General errors (WF-GN-xxx)
"""

from typing import ClassVar, Optional


# =============================================================================
# Error Code Registry
# =============================================================================


class ErrorCode:
    """Error code constants for Word Forge exceptions.

    These codes provide unique identifiers for error types, useful for
    logging, monitoring, and programmatic error handling.
    """

    # General errors (WF-GN-xxx)
    GENERAL_ERROR = "WF-GN-001"
    UNEXPECTED_ERROR = "WF-GN-002"

    # Database errors (WF-DB-xxx)
    DATABASE_ERROR = "WF-DB-001"
    DATABASE_CONNECTION_ERROR = "WF-DB-002"
    DATABASE_QUERY_ERROR = "WF-DB-003"
    DATABASE_TRANSACTION_ERROR = "WF-DB-004"

    # Graph errors (WF-GR-xxx)
    GRAPH_ERROR = "WF-GR-001"
    GRAPH_ANALYSIS_ERROR = "WF-GR-002"
    GRAPH_EXPORT_ERROR = "WF-GR-003"
    GRAPH_IMPORT_ERROR = "WF-GR-004"
    GRAPH_UPDATE_ERROR = "WF-GR-005"
    GRAPH_QUERY_ERROR = "WF-GR-006"
    GRAPH_CONNECTION_ERROR = "WF-GR-007"
    GRAPH_TRAVERSAL_ERROR = "WF-GR-008"
    GRAPH_STORAGE_ERROR = "WF-GR-009"
    GRAPH_SERIALIZATION_ERROR = "WF-GR-010"
    GRAPH_IO_ERROR = "WF-GR-011"
    GRAPH_LAYOUT_ERROR = "WF-GR-012"
    NODE_NOT_FOUND_ERROR = "WF-GR-013"
    GRAPH_DATA_ERROR = "WF-GR-014"
    GRAPH_VISUALIZATION_ERROR = "WF-GR-015"
    GRAPH_DIMENSION_ERROR = "WF-GR-016"

    # Queue errors (WF-QU-xxx)
    QUEUE_ERROR = "WF-QU-001"
    QUEUE_FULL_ERROR = "WF-QU-002"
    QUEUE_EMPTY_ERROR = "WF-QU-003"

    # Parser errors (WF-PA-xxx)
    PARSER_ERROR = "WF-PA-001"
    PARSER_RESOURCE_ERROR = "WF-PA-002"

    # Vector errors (WF-VE-xxx)
    VECTOR_ERROR = "WF-VE-001"
    VECTOR_STORAGE_ERROR = "WF-VE-002"
    VECTOR_SEARCH_ERROR = "WF-VE-003"
    VECTOR_INDEX_ERROR = "WF-VE-004"
    VECTOR_EMBEDDING_ERROR = "WF-VE-005"

    # Conversation errors (WF-CV-xxx)
    CONVERSATION_ERROR = "WF-CV-001"
    CONVERSATION_NOT_FOUND_ERROR = "WF-CV-002"
    CONVERSATION_STATE_ERROR = "WF-CV-003"


# =============================================================================
# Generic Exception Classes
# =============================================================================


class WordForgeError(Exception):
    """Base exception for all Word Forge errors.

    Attributes:
        error_code: Unique identifier for this error type
        message: Human-readable error description
        cause: Optional original exception that caused this error
    """

    error_code: ClassVar[str] = ErrorCode.GENERAL_ERROR

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        """Initialize with detailed error message and optional cause.

        Args:
            message: Error description with context
            cause: Optional; original exception that caused this error
        """
        super().__init__(message)
        self.__cause__ = cause
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Provide detailed error message including cause if available."""
        error_msg = f"[{self.error_code}] {self.message}"
        if self.cause:
            error_msg += f" | Cause: {str(self.cause)}"
        return error_msg


class DatabaseError(WordForgeError):
    """Base exception for database operations."""

    error_code: ClassVar[str] = ErrorCode.DATABASE_ERROR


class GraphError(WordForgeError):
    """Base exception for graph operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_ERROR


class GraphAnalysisError(GraphError):
    """Base exception for graph analysis operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_ANALYSIS_ERROR


class GraphExportError(GraphError):
    """Base exception for graph export operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_EXPORT_ERROR


class GraphImportError(GraphError):
    """Base exception for graph import operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_IMPORT_ERROR


class GraphUpdateError(GraphError):
    """Base exception for graph update operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_UPDATE_ERROR


class GraphQueryError(GraphError):
    """Base exception for graph query operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_QUERY_ERROR


class GraphConnectionError(GraphError):
    """Base exception for graph connection operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_CONNECTION_ERROR


class GraphTraversalError(GraphError):
    """Base exception for graph traversal operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_TRAVERSAL_ERROR


class GraphStorageError(GraphError):
    """Base exception for graph storage operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_STORAGE_ERROR


class GraphSerializationError(GraphError):
    """Base exception for graph serialization operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_SERIALIZATION_ERROR


class GraphIOError(GraphError):
    """Base exception for graph input/output operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_IO_ERROR


class GraphLayoutError(GraphError):
    """Base exception for graph layout operations."""

    error_code: ClassVar[str] = ErrorCode.GRAPH_LAYOUT_ERROR


class QueueError(WordForgeError):
    """Base exception for queue operations."""

    error_code: ClassVar[str] = ErrorCode.QUEUE_ERROR


class ParserError(WordForgeError):
    """Base exception for parser operations."""

    error_code: ClassVar[str] = ErrorCode.PARSER_ERROR


class VectorError(WordForgeError):
    """Base exception for vector operations."""

    error_code: ClassVar[str] = ErrorCode.VECTOR_ERROR


class VectorStorageError(VectorError):
    """Raised when vector storage operations fail."""

    error_code: ClassVar[str] = ErrorCode.VECTOR_STORAGE_ERROR


class VectorSearchError(VectorError):
    """Raised when vector search operations fail."""

    error_code: ClassVar[str] = ErrorCode.VECTOR_SEARCH_ERROR


class VectorIndexError(VectorError):
    """Raised when vector indexing operations fail."""

    error_code: ClassVar[str] = ErrorCode.VECTOR_INDEX_ERROR


class VectorEmbeddingError(VectorError):
    """Raised when vector embedding generation fails."""

    error_code: ClassVar[str] = ErrorCode.VECTOR_EMBEDDING_ERROR


class ConversationError(WordForgeError):
    """Base exception for conversation operations."""

    error_code: ClassVar[str] = ErrorCode.CONVERSATION_ERROR


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation lookup fails."""

    error_code: ClassVar[str] = ErrorCode.CONVERSATION_NOT_FOUND_ERROR


class ConversationStateError(ConversationError):
    """Raised when conversation state is invalid."""

    error_code: ClassVar[str] = ErrorCode.CONVERSATION_STATE_ERROR


# =============================================================================
# Graph Specific Exceptions
# =============================================================================


class NodeNotFoundError(GraphError):
    """
    Raised when a term lookup fails within the graph.

    This occurs when attempting to access a node that doesn't exist,
    typically during relationship or subgraph operations.
    """

    error_code: ClassVar[str] = ErrorCode.NODE_NOT_FOUND_ERROR


class GraphDataError(GraphError):
    """
    Raised when graph data structure contains inconsistencies.

    This indicates a structural problem with the graph data itself,
    such as missing required node attributes or invalid edge structures.
    """

    error_code: ClassVar[str] = ErrorCode.GRAPH_DATA_ERROR


class GraphVisualizationError(GraphError):
    """
    Raised when graph visualization generation fails.

    This typically occurs during rendering operations, HTML generation,
    or when visualization libraries encounter errors.
    """

    error_code: ClassVar[str] = ErrorCode.GRAPH_VISUALIZATION_ERROR


class GraphDimensionError(GraphError):
    """
    Raised when graph dimensional operations fail.

    This occurs when attempting to set invalid dimensions or
    when dimensional operations (like projection) fail.
    """

    error_code: ClassVar[str] = ErrorCode.GRAPH_DIMENSION_ERROR
