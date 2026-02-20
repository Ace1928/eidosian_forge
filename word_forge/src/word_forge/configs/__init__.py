"""
Word Forge Configuration Package.

This package provides the configuration system for Word Forge, including
type definitions, configuration management, and serialization utilities.

Modules:
    config_essentials: Core types, errors, and utilities (legacy, comprehensive)
    types/: Modularized type definitions (recommended for new code)
    config: Main configuration classes
    vector_store: Vector storage configuration

Usage:
    # Import from types/ for modular access
    from word_forge.configs.types import Error, Result, TaskPriority

    # Import from config_essentials for backward compatibility
    from word_forge.configs.config_essentials import Error, Result
"""

# Re-export key types for convenience
from .config_essentials import (
    DATA_ROOT,
    LOGS_ROOT,
    PROJECT_ROOT,
    ConfigError,
    Error,
    Result,
    serialize_config,
    serialize_dataclass,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    "Error",
    "Result",
    "ConfigError",
    "serialize_config",
    "serialize_dataclass",
]
