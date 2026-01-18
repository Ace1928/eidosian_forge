"""
Unified configuration system for Word Forge.
This module defines all exceptions used throughout the configuration
system, providing a consistent way to handle errors related to configuration
loading, parsing, and validation.
"""

from pathlib import Path
from typing import Final

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class PathError(ConfigError):
    """Raised when a path operation fails."""

    pass


class EnvVarError(ConfigError):
    """Raised when an environment variable cannot be processed."""

    pass


# Vector configuration specific errors
class VectorConfigError(ConfigError):
    """Raised when vector configuration is invalid."""

    pass


class VectorIndexError(ConfigError):
    """Raised when vector index operations fail."""

    pass


# Graph-specific error
class GraphConfigError(ConfigError):
    """Raised when graph configuration is invalid."""

    pass


# Custom error for logging configuration issues
class LoggingConfigError(ConfigError):
    """Raised when logging configuration is invalid."""

    pass


# Database-specific error types
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
