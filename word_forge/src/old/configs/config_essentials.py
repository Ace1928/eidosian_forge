"""
This module defines all essential types and constants used throughout the
configuration system, providing a consistent way to handle various
configuration-related tasks, including serialization, path management,
and error handling.
"""

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, Sequence, TypeVar, cast

from word_forge.configs.config_enums import (
    ConversationExportFormat,
    ConversationRetentionPolicy,
    DatabaseDialect,
    GraphColorScheme,
    GraphLayoutAlgorithm,
    LogDestination,
    LogFormatTemplate,
    LogRotationStrategy,
    QueuePerformanceProfile,
    StorageType,
    VectorIndexStatus,
    VectorModelType,
)
from word_forge.configs.config_exceptions import (
    ConfigError,
    DatabaseConfigError,
    DatabaseConnectionError,
    EnvVarError,
    GraphConfigError,
    LoggingConfigError,
    PathError,
    VectorConfigError,
    VectorIndexError,
)
from word_forge.configs.config_protocols import JSONSerializable
from word_forge.configs.config_types import (
    ConfigValue,
    ConnectionPoolMode,
    ConversationMetadataSchema,
    ConversationStatusMap,
    ConversationStatusValue,
    EmotionRange,
    EnvMapping,
    EnvVarType,
    GraphEdgeWeightStrategy,
    GraphExportFormat,
    GraphNodeSizeStrategy,
    InstructionTemplate,
    JsonDict,
    JsonList,
    JsonPrimitive,
    JsonValue,
    LockType,
    LogLevel,
    PathLike,
    QueueMetricsFormat,
    SampleRelationship,
    SampleWord,
    SerializedConfig,
    SQLitePragmas,
    SQLTemplates,
    TransactionIsolationLevel,
    VectorDistanceMetric,
    VectorOptimizationLevel,
    VectorSearchStrategy,
)

# ==========================================
# Generic Type Variables
# ==========================================

# Generic type parameter for configuration value types
T = TypeVar("T")

# Generic type parameter for function return types
R = TypeVar("R")

# ==========================================
# Project Paths
# ==========================================

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"


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
        d: Dict[str, ConfigValue] = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                d[key] = serialize_config(value)
        return d
    elif isinstance(obj, (list, tuple)):
        return [serialize_config(item) for item in cast(Sequence[Any], obj)]
    elif isinstance(obj, dict):
        return {
            key: serialize_config(value)
            for key, value in cast(Dict[Any, Any], obj).items()
        }
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Generic type variables
    "T",
    "R",
    # Basic JSON and configuration types
    "JsonPrimitive",
    "JsonDict",
    "JsonList",
    "JsonValue",
    "ConfigValue",
    "SerializedConfig",
    "PathLike",
    "EnvVarType",
    "EnvMapping",
    # Domain-specific types
    "EmotionRange",
    "SampleWord",
    "SampleRelationship",
    "JSONSerializable",
    # Error types
    "ConfigError",
    "PathError",
    "EnvVarError",
    # Enum types
    "StorageType",
    # Template types
    "InstructionTemplate",
    # Path constants
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # Utility functions
    "serialize_dataclass",
    "serialize_config",
    # SQL-related types
    "SQLitePragmas",
    "SQLTemplates",
    # Queue configuration types
    "LockType",
    "QueuePerformanceProfile",
    "QueueMetricsFormat",
    # Conversation configuration types
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    "ConversationRetentionPolicy",
    "ConversationExportFormat",
    # Vectorizer configuration types
    "VectorModelType",
    "VectorIndexStatus",
    "VectorSearchStrategy",
    "VectorDistanceMetric",
    "VectorOptimizationLevel",
    "VectorConfigError",
    "VectorIndexError",
    # Graph configuration types
    "GraphLayoutAlgorithm",
    "GraphColorScheme",
    "GraphExportFormat",
    "GraphNodeSizeStrategy",
    "GraphEdgeWeightStrategy",
    "GraphConfigError",
    # Logging configuration types
    "LogLevel",
    "LogFormatTemplate",
    "LogRotationStrategy",
    "LogDestination",
    "LoggingConfigError",
    # Database configuration types
    "DatabaseDialect",
    "TransactionIsolationLevel",
    "DatabaseConfigError",
    "DatabaseConnectionError",
    "ConnectionPoolMode",
]
