"""Type Forge: A module for dynamic type validation and transformation with recursive precision.

This module serves as the entry point for the Type Forge framework,
providing a unified interface for dynamic type creation, validation,
and transformation with structural integrity and recursive precision.

The TypeForge framework enables:
    - Dynamic type creation with strict validation guarantees
    - Recursive schema validation with elegant error handling
    - Type conversion with compile-time and runtime safety
    - Structural integrity verification through composable validators

Modules:
    core: Contains base classes, protocols, and exceptions for type validation
    forge: Implements the main TypeForge class and associated functionality
    validators: Provides specialized validators for comprehensive type checking
    typing: Type definitions, protocols, and type-related utilities
    utils: General utility functions for type manipulation and formatting

Classes:
    TypeForge: Main entry point for dynamic type creation and validation
    BasicValidator: Foundation validator for simple type validation
    CompositeValidator: Combines multiple validators for complex validation
    ValidationResult: Immutable result of a validation operation
    TypeViolation: Precisely describes a type violation with context
    TypeViolationKind: Enumeration of possible validation failure types
    ValidatorFactory: Creates specialized validators for diverse scenarios

Functions:
    format_validation_error(violation: TypeViolation) -> str:
        Formats type violations as human-readable error messages
    deduplicate_violations(violations: Sequence[TypeViolation]) -> List[TypeViolation]:
        Removes duplicate violations while preserving order and context
    is_valid_type_descriptor(value: Any) -> bool:
        Determines if a value can serve as a valid type descriptor

Typical usage example:
    >>> from type_forge import TypeForge
    >>> forge = TypeForge()
    >>> UserSchema = forge.create({
    ...     "name": str,
    ...     "age": int,
    ...     "email": forge.Email()
    ... })
    >>> user = UserSchema(name="Alice", age=30, email="alice@example.com")
"""

from typing import List, Sequence, TypeVar, cast

# Core components with explicit imports instead of wildcards
from .core import (
    BasicValidator,
    CompositeValidator,
    TypeForge,
    TypeViolation,
    TypeViolationKind,
    ValidationResult,
    ValidatorFactory,
)
from .typing import TypeDescriptor, TypeValidator, ValidationContext, ValidationProtocol
from .utils import (
    deduplicate_violations,
    format_validation_error,
    is_valid_type_descriptor,
)
from .validators import (
    BooleanValidator,
    DictValidator,
    ListValidator,
    NumberValidator,
    OptionalValidator,
    StringValidator,
    UnionValidator,
)

# Define explicit type for __all__ to ensure it's recognized as a list of exported names
__all__: List[str] = [
    # Core components
    "TypeForge",
    "BasicValidator",
    "CompositeValidator",
    "ValidationResult",
    "TypeViolation",
    "TypeViolationKind",
    "ValidatorFactory",
    # Type definitions and protocols
    "TypeDescriptor",
    "TypeValidator",
    "ValidationProtocol",
    "ValidationContext",
    # Utility functions
    "format_validation_error",
    "deduplicate_violations",
    "is_valid_type_descriptor",
    # Specialized validators
    "StringValidator",
    "NumberValidator",
    "BooleanValidator",
    "ListValidator",
    "DictValidator",
    "OptionalValidator",
    "UnionValidator",
]

# Type variable for generic operations
T = TypeVar("T")
