"""
version_forge: Semantic versioning with cross-component compatibility enforcement.

A modular toolkit for managing version information, compatibility checking,
and automatic version updates across codebases and components.
"""
from .cli import main as cli
from .compatibility.matrix import CompatibilityMatrix
from .compatibility.validator import DependencyValidator
from .core.config import VersionConfig
from .core.version import (DEFAULT_VERSION, SimpleVersion, format_version,
                           parse_version)
from .operations.compare import calculate_delta, is_compatible
from .operations.migration import MigrationGuideGenerator
from .operations.update import update_version

# Version of the version_forge itself
__version__ = "3.14.7"

# Public API
__all__ = [
    "SimpleVersion",
    "parse_version",
    "format_version",
    "DEFAULT_VERSION",
    "VersionConfig",
    "CompatibilityMatrix",
    "DependencyValidator",
    "calculate_delta",
    "is_compatible",
    "update_version",
    "MigrationGuideGenerator",
    "cli"
]
