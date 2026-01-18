"""
Type Definitions for the Type Forge system.

"""

from __future__ import annotations

from enum import Enum
from typing import final

from . import __version__  # noqa: F401

# ──────────────────────────────────────────────────────────────
# Core Type Variables and Aliases
# ──────────────────────────────────────────────────────────────


version = __version__


# ──────────────────────────────────────────────────────────────
# Type Definitions
# ──────────────────────────────────────────────────────────────


@final
class TypeCategory(Enum):
    """
    Categorization of type structures for semantic operations.

    This enumeration provides a richer vocabulary for describing type
    relationships beyond basic inheritance, allowing for precise categorization
    of types based on their structural and behavioral properties.

    Attributes:
        ATOMIC: Primitive types that are indivisible units (int, float, bool)
        COMPOSITE: Types composed of other types (classes, dataclasses)
        STRUCTURAL: Types that define a structure or shape (NamedTuple, Struct)
        CONTAINER: Types that hold collections of other values (List, Dict)
        FUNCTION: Types representing callable objects (functions, lambdas)
        PROTOCOL: Types representing interfaces or behaviors (Protocols)
        GENERIC: Types with type parameters (List[T], Dict[K, V])
        SPECIAL: Types with special behaviors (Optional, Union)
        NUMERIC: Types representing numbers (int, float, complex)
        TEXT: Types representing text (str, bytes)
        BOOLEAN: Types representing boolean values (bool)
        DATETIME: Types representing dates and times
        ENUM: Enumeration types
        BINARY: Binary data types (bytes, bytearray)
        NETWORK: Network-related types (URL, IP address)
        RECURSIVE: Self-referential types

    Examples:
        >>> TypeCategory.ATOMIC.name
        'ATOMIC'
        >>> TypeCategory.CONTAINER.value
        'container'
        >>> str(TypeCategory.FUNCTION)
        'function'
    """

    ATOMIC = "atomic"
    COMPOSITE = "composite"
    STRUCTURAL = "structural"
    CONTAINER = "container"
    FUNCTION = "function"
    PROTOCOL = "protocol"
    GENERIC = "generic"
    SPECIAL = "special"
    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ENUM = "enum"
    BINARY = "binary"
    NETWORK = "network"
    RECURSIVE = "recursive"

    def __str__(self) -> str:
        """
        String representation of the type category.

        Returns:
            str: The name of the type category in lowercase

        Examples:
            >>> str(TypeCategory.ATOMIC)
            'atomic'
            >>> str(TypeCategory.COMPOSITE)
            'composite'
        """
        return self.value


@final
class ValidationLevel(Enum):
    """
    Levels of validation strictness for type validation functions.

    This enumeration provides different levels of strictness for
    type validation operations, allowing for flexible type checking
    based on application requirements.

    Attributes:
        STRICT: Allow no type variance, exact type match required.
            Types must be identical (A is A, not A is subclass of B).
        STANDARD: Allow normal subtype relationships (inheritance).
            Types can be subclasses (B is acceptable when A is required if B inherits
            from A).
        PERMISSIVE: Allow type conversion where possible.
            Attempts to convert between compatible types (str to int if str contains
            a number).
        DYNAMIC: Use duck typing and runtime checks.
            Checks for attribute/method presence rather than type identity.
        STRUCTURAL: Check structural compatibility only.
            Types are compatible if they have compatible structures regardless
            of inheritance.
        COVARIANT: Allow covariant substitution.
            A type B can be used where A is required if B is a subtype of A.
        CONTRAVARIANT: Allow contravariant substitution.
            A type B can be used where A is required if A is a subtype of B.
        NONE: No validation performed.
            All types are accepted without verification (use with caution).

    Examples:
        >>> ValidationLevel.STRICT.name
        'STRICT'
        >>> ValidationLevel.STANDARD.value
        'standard'
        >>> str(ValidationLevel.PERMISSIVE)
        'permissive'
    """

    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    DYNAMIC = "dynamic"
    STRUCTURAL = "structural"
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    NONE = "none"

    def __str__(self) -> str:
        """
        String representation of the validation level.

        Returns:
            str: The name of the validation level in lowercase

        Examples:
            >>> str(ValidationLevel.STRICT)
            'strict'
            >>> str(ValidationLevel.STANDARD)
            'standard'
        """
        return self.value


@final
class TypeCompatibility(Enum):
    """
    Classification of type compatibility relationships for conversion operations.

    This enumeration provides a detailed classification system for describing
    the compatibility between two types, guiding conversion strategies and
    validation processes.

    Attributes:
        IDENTICAL: Types are exactly the same (int and int).
        SUBTYPE: Source is a subtype of target (bool is a subtype of int).
        SUPERTYPE: Target is a subtype of source (int is a supertype of bool).
        CONVERTIBLE: Types can be converted explicitly (str to int).
        STRUCTURALLY_COMPATIBLE: Types share compatible structures (NamedTuple vs dataclass).
        PROTOCOL_COMPATIBLE: Source satisfies target's protocol requirements.
        CONTAINER_COMPATIBLE: Container types with compatible element types.
        IMPLICIT_CONVERTIBLE: Types can be converted implicitly (int to float).
        INCOMPATIBLE: Types cannot be converted or are fundamentally different.

    Examples:
        >>> TypeCompatibility.IDENTICAL.name
        'IDENTICAL'
        >>> TypeCompatibility.CONVERTIBLE.value
        'convertible'
        >>> str(TypeCompatibility.INCOMPATIBLE)
        'incompatible'
    """

    IDENTICAL = "identical"
    SUBTYPE = "subtype"
    SUPERTYPE = "supertype"
    CONVERTIBLE = "convertible"
    STRUCTURALLY_COMPATIBLE = "structurally_compatible"
    PROTOCOL_COMPATIBLE = "protocol_compatible"
    CONTAINER_COMPATIBLE = "container_compatible"
    IMPLICIT_CONVERTIBLE = "implicit_convertible"
    INCOMPATIBLE = "incompatible"

    def __str__(self) -> str:
        """
        String representation of the type compatibility.

        Returns:
            str: The name of the type compatibility in lowercase

        Examples:
            >>> str(TypeCompatibility.IDENTICAL)
            'identical'
            >>> str(TypeCompatibility.CONVERTIBLE)
            'convertible'
        """
        return self.value

    def is_compatible(self) -> bool:
        """
        Determine if this compatibility level allows conversion.

        Returns:
            bool: True if types are compatible (can be converted), False otherwise

        Examples:
            >>> TypeCompatibility.IDENTICAL.is_compatible()
            True
            >>> TypeCompatibility.INCOMPATIBLE.is_compatible()
            False
        """
        return self != TypeCompatibility.INCOMPATIBLE


@final
class ValidationSeverity(Enum):
    """
    Severity levels for validation errors and warnings.

    This enumeration provides different severity levels for validation
    issues, enabling appropriate handling and reporting based on criticality.

    Attributes:
        FATAL: Critical error that must be fixed and prevents operation.
        ERROR: Serious issue that should be fixed but might allow partial operation.
        WARNING: Potential issue or deviation from best practice.
        INFO: Informational message about validation results.
        DEBUG: Technical details useful for debugging validation.

    Examples:
        >>> ValidationSeverity.ERROR.name
        'ERROR'
        >>> ValidationSeverity.WARNING.value
        'warning'
        >>> str(ValidationSeverity.INFO)
        'info'
    """

    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

    def __str__(self) -> str:
        """
        String representation of the validation severity.

        Returns:
            str: The name of the validation severity in lowercase

        Examples:
            >>> str(ValidationSeverity.ERROR)
            'error'
            >>> str(ValidationSeverity.WARNING)
            'warning'
        """
        return self.value

    def is_error(self) -> bool:
        """
        Check if this severity level represents an error condition.

        Returns:
            bool: True if this severity is an error or fatal error

        Examples:
            >>> ValidationSeverity.FATAL.is_error()
            True
            >>> ValidationSeverity.WARNING.is_error()
            False
        """
        return self in (ValidationSeverity.ERROR, ValidationSeverity.FATAL)

    def is_blocker(self) -> bool:
        """
        Check if this severity level should block operation.

        Returns:
            bool: True if this severity should prevent operation

        Examples:
            >>> ValidationSeverity.FATAL.is_blocker()
            True
            >>> ValidationSeverity.ERROR.is_blocker()
            False
        """
        return self == ValidationSeverity.FATAL
