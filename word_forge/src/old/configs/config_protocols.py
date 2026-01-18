"""
Configuration protocols and interfaces for Word Forge.

This module defines the protocol interfaces that configuration components
must implement to ensure consistent functionality across the Word Forge system.
It establishes contracts for configuration components and serialization behavior
without dictating implementation details.

Protocol architecture:
    - Type-safe configuration components with consistent interfaces
    - Environment variable overriding capability across all components
    - Standardized JSON serialization for configuration objects
"""

from pathlib import Path
from typing import ClassVar, Dict, Final, Protocol, Tuple, TypeVar

from word_forge.configs.config_types import EnvVarType

# ==========================================
# Project Paths
# ==========================================

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"

# ==========================================
# Protocol Type Variables
# ==========================================

# Type variable bound to ConfigComponent protocol for generic configuration handling
C = TypeVar("C", bound="ConfigComponent")


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


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Project paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # Type variables
    "C",
    # Protocol classes
    "ConfigComponent",
    "JSONSerializable",
]
