"""Core module for the TypeForge type validation system.

This module provides the essential components for runtime type validation,
dynamic type creation, and structural type checking. It forms the backbone
of the TypeForge framework, enabling precise control over data structures
with minimal runtime overhead.

The components in this module follow composition patterns with strict
typing to ensure type safety throughout the validation pipeline. Each
component has a single responsibility and communicates through well-defined
interfaces.

Architecture:
    TypeForgeBase: The root class establishing the type identity system
    BaseValidator: Validation interface with pure functional core
    ValidationResult: Immutable result structure with strict guarantees
    Exceptions: Specialized error hierarchy for precise error handling

Exports:
    BaseValidator: Abstract base class defining the validator interface
        with validate() and is_valid() methods
    TypeForgeBase: Root class for all TypeForge objects providing common
        functionality and type identity
    ValidationResult: Immutable container for validation outcomes with
        status, value, and violation details
    ValidationError: Exception raised when validation fails with
        detailed violation information
    TypeForgeException: Base exception class for all TypeForge errors
        to enable specific error handling
    TypeCreationError: Exception raised when type construction fails
        due to invalid parameters or configuration
    ConfigurationError: Exception for invalid TypeForge configuration
        typically raised during initialization
    TypeViolation: Detailed information about a specific type violation
        including path, value, and violation kind
    TypeViolationKind: Enumeration of possible violation categories
        (e.g., type mismatch, missing field, extra field)

Example:
    >>> from type_forge.core import ValidationResult, TypeForgeBase, TypeViolation, TypeViolationKind
    >>>
    >>> # Create a validation result for a successful validation
    >>> result = ValidationResult(valid=True, value=42, violations=[])
    >>> assert isinstance(result, TypeForgeBase)
    >>>
    >>> # Example with validation failure
    >>> violations = [TypeViolation(
    ...     path="age",
    ...     expected="int",
    ...     received="str",
    ...     value="twenty",
    ...     kind=TypeViolationKind.TYPE_MISMATCH
    ... )]
    >>> failed_result = ValidationResult(valid=False, value=None, violations=violations)
    >>> assert not failed_result.valid
    >>>
    >>> # Accessing violation details
    >>> if not failed_result.valid:
    ...     for violation in failed_result.violations:
    ...         print(f"Error at {violation.path}: Expected {violation.expected, got {violation.received}")

Notes:
    All components maintain strict immutability to ensure thread safety and
    prevent side-effects during validation operations.

See Also:
    type_forge.validators: Higher-level validation components
    type_forge.types: Type definition and construction utilities
"""

from __future__ import annotations

# Standard library imports
import sys
from typing import Final, List, TypeVar

# Local imports, organized by component category
from .base import BaseValidator, TypeForgeBase, ValidationResult
from .exceptions import (
    ConfigurationError,
    TypeCreationError,
    TypeForgeException,
    TypeViolation,
    TypeViolationKind,
    ValidationError,
)

# Type definitions
# TypeVar representing any TypeForgeBase subclass for generic operations
T_ForgeType = TypeVar("T_ForgeType", bound=TypeForgeBase)

# Module version information
__version__: Final[str] = "1.0.0"  # Follows semantic versioning

# Module exports with strict typing - frozen at import time
__all__: Final[List[str]] = [
    "BaseValidator",
    "TypeForgeBase",
    "ValidationResult",
    "ValidationError",
    "TypeForgeException",
    "TypeCreationError",
    "ConfigurationError",
    "TypeViolation",
    "TypeViolationKind",
]

# Runtime environment verification
if sys.version_info < (3, 8):  # pragma: no cover
    raise RuntimeError(
        f"TypeForge requires Python 3.8+ (current: {sys.version_info.major}.{sys.version_info.minor})"
    )
