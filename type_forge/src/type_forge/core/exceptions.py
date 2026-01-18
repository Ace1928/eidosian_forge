"""Exception hierarchy and type violation tracking for TypeForge.

This module provides a comprehensive exception system for the TypeForge library,
enabling precise error reporting, structured violation tracking, and clear
error hierarchies that maintain the principle of informative failure.

Classes:
    TypeForgeException: Base exception for all TypeForge errors.
    ValidationError: Raised when validation constraints aren't met.
    TypeCreationError: Raised when type construction fails.
    ConfigurationError: Raised when configuration is invalid.
    TypeViolationKind: Enumeration of violation categories.
    TypeViolation: Immutable record of a type violation with path tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal  # For precise string literal typing


class TypeForgeException(Exception):
    """Base class for all exceptions raised by the TypeForge module.

    All exceptions in this library inherit from this class, enabling
    targeted exception handling for TypeForge-specific errors.

    Examples:
        ```python
        try:
            # Some TypeForge operation
            pass
        except TypeForgeException as e:
            # Handle any TypeForge-related error
            pass
        ```
    """


class ValidationError(TypeForgeException):
    """Exception raised for data validation errors.

    Raised when data fails to meet validation constraints defined
    in a type schema or validation rule.

    Args:
        message: Descriptive error message explaining the validation failure.

    Examples:
        ```python
        raise ValidationError("Age must be greater than 0")
        ```
    """

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Detailed description of the validation error.
        """
        super().__init__(message)


class TypeCreationError(TypeForgeException):
    """Exception raised for errors during type creation.

    Raised when attempting to create a type definition fails due to
    invalid parameters, conflicting constraints, or other type
    construction issues.

    Args:
        message: Descriptive error message explaining the type creation failure.

    Examples:
        ```python
        raise TypeCreationError("Cannot create recursive type without base case")
        ```
    """

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Detailed description of the type creation error.
        """
        super().__init__(message)


class ConfigurationError(TypeForgeException):
    """Exception raised for configuration-related errors.

    Raised when TypeForge is configured with invalid, incompatible,
    or missing configuration parameters.

    Args:
        message: Descriptive error message explaining the configuration issue.

    Examples:
        ```python
        raise ConfigurationError("Invalid serialization format specified")
        ```
    """

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Detailed description of the configuration error.
        """
        super().__init__(message)


# Using Literal types for precise string typing
ViolationKindLiteral = Literal[
    "wrong_type", "missing_key", "invalid_value", "schema_mismatch", "conversion_error"
]


class TypeViolationKind(Enum):
    """Enumeration of possible type violation categories.

    Provides a structured taxonomy of type violations for precise
    error categorization and handling.

    Attributes:
        WRONG_TYPE: Value has incorrect type.
        MISSING_KEY: Required key is absent.
        INVALID_VALUE: Value fails validation constraints.
        SCHEMA_MISMATCH: Value structure doesn't match schema.
        CONVERSION_ERROR: Type conversion failed.

    Examples:
        ```python
        if isinstance(value, str):
            return TypeViolationKind.WRONG_TYPE
        ```
    """

    WRONG_TYPE = "wrong_type"
    MISSING_KEY = "missing_key"
    INVALID_VALUE = "invalid_value"
    SCHEMA_MISMATCH = "schema_mismatch"
    CONVERSION_ERROR = "conversion_error"

    def __str__(self) -> str:
        """Generate string representation of the violation kind.

        Returns:
            The string value of the enumeration.
        """
        return self.value


@dataclass(frozen=True)
class TypeViolation:
    """Immutable record of a type violation with path tracking.

    Provides a structured representation of a type violation with
    context information for precise error reporting and diagnosis.
    The frozen dataclass ensures immutability for safer error handling.

    Attributes:
        path: JSON path to the location of the violation.
        expected: Description of expected type or value.
        found: Description of actual type or value found.
        kind: Category of violation from TypeViolationKind.

    Examples:
        ```python
        violation = TypeViolation(
            path="user.address.zipcode",
            expected="string of 5 digits",
            found="'ABC123'",
            kind=TypeViolationKind.INVALID_VALUE
        )
        ```
    """

    path: str
    expected: str
    found: str
    kind: TypeViolationKind

    def __str__(self) -> str:
        """Generate human-readable representation of the violation.

        Returns:
            Formatted string with violation details including path,
            expected value, found value, and violation kind.
        """
        return (
            f"At path '{self.path}': Expected {self.expected}, "
            f"found {self.found} ({self.kind.value})"
        )
