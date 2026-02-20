"""
Error handling types for Word Forge configuration system.

This module provides the Result pattern and Error types for exception-free
error handling across component boundaries.

Classes:
    ErrorCategory: Categories of errors for systematic handling
    ErrorSeverity: Severity levels for errors
    Error: Immutable error object with comprehensive context
    Result: Monadic result type for error handling without exceptions
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    cast,
)

try:
    from eidosian_core import eidosian
except ImportError:

    def eidosian():
        """Fallback decorator when eidosian_core is not available."""

        def decorator(func):
            return func

        return decorator


# Type variables for Result
T = TypeVar("T")
R = TypeVar("R")


class ErrorCategory(Enum):
    """Categories of errors for systematic handling and reporting."""

    VALIDATION = auto()  # Input validation failures
    RESOURCE = auto()  # Resource availability issues
    BUSINESS = auto()  # Business rule violations
    EXTERNAL = auto()  # External system failures
    UNEXPECTED = auto()  # Unexpected failures
    CONFIGURATION = auto()  # Configuration errors
    SECURITY = auto()  # Security-related issues


class ErrorSeverity(Enum):
    """Severity levels for errors to guide handling strategies."""

    FATAL = auto()  # System cannot continue operation
    ERROR = auto()  # Operation failed completely
    WARNING = auto()  # Operation completed with issues
    INFO = auto()  # Operation completed with non-critical adjustments


@dataclass(frozen=True)
class Error:
    """
    Immutable error object with comprehensive context for accurate diagnostics.

    Provides a unified structure for error handling across the system,
    with severity classification, error codes, and contextual information.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        category: Error category for classification
        severity: Error severity for handling strategy
        context: Dictionary of additional contextual information
        trace: Optional stack trace for debugging
    """

    message: str
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, str] = field(default_factory=dict)
    trace: Optional[str] = None

    @classmethod
    def create(
        cls,
        message: str,
        code: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[Dict[str, str]] = None,
    ) -> "Error":
        """
        Factory method for creating errors with standardized formatting.

        Args:
            message: Human-readable error description
            code: Machine-readable error code
            category: Error category for classification
            severity: Error severity for handling strategy
            context: Optional dictionary of contextual information

        Returns:
            Fully initialized Error object
        """
        import traceback

        return cls(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context or {},
            trace=traceback.format_exc(),
        )


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    Monadic result type for error handling without exceptions.

    Implements the Result pattern for expressing success/failure
    without raising exceptions across component boundaries.

    Attributes:
        value: The success value (None if error)
        error: The error details (None if success)
    """

    value: Optional[T] = None
    error: Optional[Error] = None

    @property
    def is_success(self) -> bool:
        """Determine if this is a successful result."""
        return self.error is None

    @property
    def is_failure(self) -> bool:
        """Determine if this is a failed result."""
        return not self.is_success

    @eidosian()
    def unwrap(self) -> T:
        """
        Extract the success value, raising an exception if this is an error result.

        Returns:
            The contained success value

        Raises:
            ValueError: If this is an error result
        """
        if not self.is_success:
            error_msg = f"Cannot unwrap failed result: {self.error.message if self.error else 'Unknown error'}"
            raise ValueError(error_msg)
        return cast(T, self.value)

    @eidosian()
    def unwrap_or(self, default: T) -> T:
        """
        Extract the success value or return a default if this is an error result.

        Args:
            default: The default value to return if this is an error result

        Returns:
            The contained success value or the provided default
        """
        if not self.is_success:
            return default
        return cast(T, self.value)

    @eidosian()
    def map(self, f: Callable[[T], R]) -> "Result[R]":
        """
        Apply a function to the value if present, otherwise pass through error.

        Args:
            f: Function to apply to the success value

        Returns:
            A new Result containing either the transformed value or the original error
        """
        if not self.is_success:
            return cast(Result[R], Result(error=self.error))
        return Result(value=f(cast(T, self.value)))

    @eidosian()
    def flat_map(self, f: Callable[[T], "Result[R]"]) -> "Result[R]":
        """
        Apply a function that returns a Result, flattening the result.

        Args:
            f: Function returning a Result to apply to the success value

        Returns:
            The Result returned by the function, or the original error
        """
        if not self.is_success:
            return cast(Result[R], Result(error=self.error))
        return f(cast(T, self.value))

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """
        Create a successful result with the given value.

        Args:
            value: The success value

        Returns:
            A successful Result containing the value
        """
        return cls(value=value)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        context: Optional[Dict[str, str]] = None,
        category: ErrorCategory = ErrorCategory.UNEXPECTED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> "Result[T]":
        """
        Create a failure result with the given error details.

        Args:
            code: Machine-readable error code
            message: Human-readable error message
            context: Optional dictionary of contextual information
            category: Error category (default: UNEXPECTED)
            severity: Error severity (default: ERROR)

        Returns:
            A failure Result containing the error details
        """
        error = Error.create(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context,
        )
        return cls(error=error)


__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "Error",
    "Result",
]
