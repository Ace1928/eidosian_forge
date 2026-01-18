"""
Vectorization configuration system for Word Forge.

This module defines the configuration schema for vector embedding models,
index storage, search parameters, and query templates used throughout
the Word Forge system for semantic similarity operations.

Architecture:
    ┌─────────────────────┐
    │  VectorizerConfig   │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Model│Index│Store│Template│Query│
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Set, TypeVar, Union, cast

from word_forge.configs.config_essentials import (
    DATA_ROOT,
    InstructionTemplate,
    StorageType,
    VectorConfigError,
    VectorDistanceMetric,
    VectorIndexError,
    VectorModelType,
    VectorOptimizationLevel,
    VectorSearchStrategy,
)
from word_forge.configs.config_types import (
    EnvMapping,
    QueryType,
    SQLQueryType,
    TemplateDict,
)

# Type variable for the clone method
T = TypeVar("T", bound="VectorizerConfig")


@dataclass(frozen=True)
class VectorizerConfig:
    """
    Vector store configuration for semantic search and similarity.

    Controls embedding models, storage parameters, indexing settings,
    and query templates for the vector storage system that powers
    semantic search and similarity operations.

    Attributes:
        model_name: Name of the embedding model to use
        model_type: Type of embedding model (transformer, sentence, etc.)
        dimension: Optional dimension override for embeddings
        index_path: Path for vector index storage
        storage_type: Storage strategy (memory or persistent disk)
        collection_name: Optional collection name for vector store
        sql_templates: SQL query templates for content retrieval
        instruction_templates: Instruction templates for different query types
        optimization_level: Tradeoff between query speed and accuracy
        search_strategy: Search algorithm strategy (exact or approximate)
        distance_metric: Distance metric for similarity calculations
        batch_size: Batch size for vector operations
        max_retries: Maximum number of retries for failed operations
        failure_cooldown_seconds: Cooldown period between retries
        enable_compression: Whether to compress vectors for storage efficiency
        compression_ratio: Compression ratio when compression is enabled
        reserved_memory_mb: Memory reserved for vector operations in MB
        enable_multi_dim: Whether to enable multi-dimensional vector spaces
        dim_separation: Z-axis separation for multi-dimensional visualization

    Usage:
        from word_forge.config import config

        # Get embedding model information
        model_name = config.vectorizer.model_name
        dimension = config.vectorizer.dimension or 768  # Default if not specified

        # Get vector index path
        index_path = config.vectorizer.get_index_path()

        # Get search template for a specific query type
        search_template = config.vectorizer.get_template("search")
    """

    # Vector embedding model configuration
    model_name: str = "intfloat/multilingual-e5-large-instruct"
    model_type: VectorModelType = VectorModelType.TRANSFORMER
    dimension: Optional[int] = None  # None = use model's default dimension
    enable_compression: bool = False
    compression_ratio: float = 0.5  # Only used when compression is enabled

    # Index and storage configuration
    index_path: str = str(DATA_ROOT / "vector_indices")
    storage_type: StorageType = StorageType.DISK
    collection_name: Optional[str] = None  # None = derive from model_name

    # Search parameter configuration
    optimization_level: VectorOptimizationLevel = "balanced"
    search_strategy: VectorSearchStrategy = "approximate"
    distance_metric: VectorDistanceMetric = "cosine"

    # Performance configuration
    batch_size: int = 32
    reserved_memory_mb: int = 512
    max_retries: int = 3
    failure_cooldown_seconds: float = 60.0

    # Multi-dimensional vector space settings
    enable_multi_dim: bool = False
    dim_separation: float = 50.0  # For multi-dimensional vector spaces

    # SQL query templates for content retrieval
    sql_templates: Dict[SQLQueryType, str] = field(
        default_factory=lambda: {
            "get_term_by_id": """
                SELECT term, definition FROM words WHERE id = ?
            """,
            "get_message_text": """
                SELECT text, speaker, conversation_id, timestamp
                FROM conversation_messages WHERE id = ?
            """,
        }
    )

    # Instruction templates for different query types
    instruction_templates: Dict[QueryType, TemplateDict] = field(
        default_factory=lambda: {
            "search": {
                "task": "Given a web search query, retrieve relevant passages that answer the query",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
            "definition": {
                "task": "Find the definition that best matches this term",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
            "similarity": {
                "task": "Measure semantic similarity between these text passages",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
        }
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_VECTOR_MODEL": ("model_name", str),
        "WORD_FORGE_VECTOR_MODEL_TYPE": ("model_type", VectorModelType),
        "WORD_FORGE_VECTOR_DIMENSION": ("dimension", int),
        "WORD_FORGE_VECTOR_INDEX_PATH": ("index_path", str),
        "WORD_FORGE_VECTOR_STORAGE_TYPE": ("storage_type", StorageType),
        "WORD_FORGE_VECTOR_BATCH_SIZE": ("batch_size", int),
        "WORD_FORGE_VECTOR_STRATEGY": ("search_strategy", str),
        "WORD_FORGE_VECTOR_METRIC": ("distance_metric", str),
        "WORD_FORGE_VECTOR_COMPRESSION": ("enable_compression", bool),
        "WORD_FORGE_VECTOR_MULTI_DIM": ("enable_multi_dim", bool),
    }

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def get_index_path(self) -> Path:
        """
        Get index path as a Path object with proper validation.

        Returns:
            Path: Object representing the vector index location

        Raises:
            VectorIndexError: If parent directory does not exist
        """
        path = Path(self.index_path)
        parent_dir = path.parent

        if not parent_dir.exists():
            raise VectorIndexError(f"Parent directory does not exist: {parent_dir}")

        return path

    @cached_property
    def effective_collection_name(self) -> str:
        """
        Get effective collection name, deriving from model name if not specified.

        Returns:
            str: Collection name to use for vector storage
        """
        if self.collection_name:
            return self.collection_name

        # Derive from model name by extracting last component and sanitizing
        model_parts = self.model_name.split("/")
        derived_name = model_parts[-1].replace("-", "_").lower()
        return f"collection_{derived_name}"

    @cached_property
    def effective_dimension(self) -> Optional[int]:
        """
        Get effective dimension, validating if specified.

        Returns:
            Optional[int]: Dimension to use for vectors, or None to use model default

        Raises:
            VectorConfigError: If dimension is invalid (not positive)
        """
        if self.dimension is not None and self.dimension <= 0:
            raise VectorConfigError(
                f"Vector dimension must be positive: {self.dimension}"
            )
        return self.dimension

    @cached_property
    def supported_template_types(self) -> Set[str]:
        """
        Get set of supported template types.

        Returns:
            Set[str]: Set of template type names available in configuration
        """
        return set(self.instruction_templates.keys())

    # ==========================================
    # Public Methods
    # ==========================================

    def get_template(self, template_type: Union[str, QueryType]) -> InstructionTemplate:
        """
        Get strongly typed instruction template for a query type.

        Args:
            template_type: The type of template to retrieve (search, definition, similarity)

        Returns:
            InstructionTemplate: Structured template with proper typing

        Raises:
            VectorConfigError: If template type doesn't exist in configuration
        """
        if template_type not in self.instruction_templates:
            raise VectorConfigError(
                f"Unknown template type: {template_type}. "
                f"Available types: {', '.join(self.supported_template_types)}"
            )

        template_dict = self.instruction_templates[template_type]
        return InstructionTemplate(
            task=template_dict["task"] or "",
            query_prefix=template_dict["query_prefix"] or "",
            document_prefix=template_dict["document_prefix"],
        )

    def get_sql_template(self, template_name: SQLQueryType) -> str:
        """
        Get SQL template by name with validation.

        Args:
            template_name: Name of the SQL template to retrieve

        Returns:
            str: SQL template string

        Raises:
            VectorConfigError: If template name doesn't exist in configuration
        """
        if template_name not in self.sql_templates:
            raise VectorConfigError(
                f"Unknown SQL template: {template_name}. "
                f"Available templates: {', '.join(self.sql_templates.keys())}"
            )
        return self.sql_templates[template_name]

    def clone(self: T, **kwargs: Any) -> T:
        """
        Create a new configuration with specified overrides.

        This is a more elegant alternative to having multiple with_* methods.

        Args:
            **kwargs: Configuration parameters to override

        Returns:
            VectorizerConfig: New configuration instance with specified parameters

        Example:
            # Create configuration optimized for speed
            speed_config = config.vectorizer.clone(
                optimization_level="speed",
                search_strategy="approximate",
                batch_size=64
            )
        """
        # Get all current field values
        current_values = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if not field_name.startswith("_") and field_name != "ENV_VARS"
        }

        # Update with provided overrides
        current_values.update(kwargs)

        # Create new instance
        return cast(T, self.__class__(**current_values))

    def with_model(
        self, model_name: str, dimension: Optional[int] = None
    ) -> "VectorizerConfig":
        """
        Create a new configuration with a different model.

        Args:
            model_name: New embedding model name
            dimension: Optional dimension override

        Returns:
            VectorizerConfig: New configuration instance

        Note:
            This method is maintained for backward compatibility.
            Consider using `clone()` for new code.
        """
        return self.clone(model_name=model_name, dimension=dimension)

    def with_storage_type(self, storage_type: StorageType) -> "VectorizerConfig":
        """
        Create a new configuration with a different storage type.

        Args:
            storage_type: New storage type (memory or disk)

        Returns:
            VectorizerConfig: New configuration instance

        Note:
            This method is maintained for backward compatibility.
            Consider using `clone()` for new code.
        """
        return self.clone(storage_type=storage_type)

    def optimize_for_performance(self, is_speed_critical: bool) -> "VectorizerConfig":
        """
        Create a new configuration optimized for either speed or accuracy.

        Args:
            is_speed_critical: Whether speed is more important than accuracy

        Returns:
            VectorizerConfig: New configuration instance with optimized settings
        """
        settings = {
            "optimization_level": "speed" if is_speed_critical else "accuracy",
            "search_strategy": "approximate" if is_speed_critical else "hybrid",
        }

        # Only increase batch size if optimizing for speed
        if is_speed_critical:
            settings["batch_size"] = self.batch_size * 2

        return self.clone(**settings)

    def get_performance_settings(self) -> Dict[str, Any]:
        """
        Get performance-related settings as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of performance settings
        """
        return {
            "batch_size": self.batch_size,
            "optimization_level": self.optimization_level,
            "search_strategy": self.search_strategy,
            "distance_metric": self.distance_metric,
            "enable_compression": self.enable_compression,
            "compression_ratio": (
                self.compression_ratio if self.enable_compression else None
            ),
            "reserved_memory_mb": self.reserved_memory_mb,
            "enable_multi_dim": self.enable_multi_dim,
            "dim_separation": self.dim_separation if self.enable_multi_dim else None,
        }

    def validate(self) -> None:
        """
        Validate the entire configuration for consistency and correctness.

        Performs comprehensive validation of all configuration parameters
        to ensure they meet constraints and logical requirements.

        Raises:
            VectorConfigError: If any validation fails with detailed error message
        """
        errors = []

        # Validate model settings
        if not self.model_name:
            errors.append("Model name cannot be empty")

        # Validate dimension if provided
        if self.dimension is not None and self.dimension <= 0:
            errors.append(f"Vector dimension must be positive: {self.dimension}")

        # Validate compression ratio
        if self.enable_compression and (
            self.compression_ratio <= 0 or self.compression_ratio >= 1
        ):
            errors.append(
                f"Compression ratio must be between 0 and 1: {self.compression_ratio}"
            )

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive: {self.batch_size}")

        # Validate multi-dimensional settings
        if self.enable_multi_dim and self.dim_separation <= 0:
            errors.append(
                f"Dimension separation must be positive: {self.dim_separation}"
            )

        # Validate templates
        for template_type, template in self.instruction_templates.items():
            if not template.get("task"):
                errors.append(f"Template '{template_type}' missing task description")
            if not template.get("query_prefix"):
                errors.append(f"Template '{template_type}' missing query prefix")

        if errors:
            raise VectorConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Configuration classes
    "VectorizerConfig",
    # Type definitions
    "VectorModelType",
    "VectorDistanceMetric",
    "VectorSearchStrategy",
    "VectorOptimizationLevel",
    # Constants
    "StorageType",
    "QueryType",
    "SQLQueryType",
    # Error types
    "VectorConfigError",
    "VectorIndexError",
]
