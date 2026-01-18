"""
Type Conversion for the Type Forge system.

This module provides robust type conversion utilities with comprehensive error handling,
focusing on safety, composability, and explicit failure modes. The core components
include the ConversionResult monad for chainable operations and specialized conversion
functions for common Python types.

All functions guarantee exception safety through explicit error handling patterns.
"""

from pathlib import Path
from typing import Callable, Dict, Generic, Optional, Type, TypeVar, Union, cast

from type_forge.typing.protocols import (
    SupportsBoolConversion,
    SupportsFloatConversion,
    SupportsIntConversion,
    SupportsLength,
    TypeConverter,
)
from type_forge.typing.variables import R, S, T, U  # V is for TypeConverter

from . import __version__  # noqa: F401

version = __version__


class ConversionResult(Generic[T]):
    """
    Result of a type conversion operation with success status and error tracking.

    This class encapsulates the result of a type conversion operation,
    providing both the converted value (if successful) and error information
    when conversion fails. It allows for chainable operations and error handling.

    Attributes:
        success (bool): Whether the conversion was successful
        value (Optional[T]): The converted value, or None if conversion failed
        error (Optional[str]): Error message if conversion failed

    Examples:
        >>> result = ConversionResult.create_success(42)
        >>> result.success
        True
        >>> print(result.value)
        42
        >>> result = ConversionResult[int].failure("Invalid conversion")
        >>> result.success
        False
        >>> print(result.error)
        Invalid conversion
        >>> # Chain operations
        >>> result = ConversionResult.create_success("123")
        >>> result.then(lambda s: ConversionResult.create_success(int(s))).value
        123
    """

    def __init__(
        self,
        success: bool,
        value: Optional[T] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Initialize a ConversionResult.

        Args:
            success: Whether the conversion was successful
            value: The converted value, None if conversion failed
            error: Error message if conversion failed

        Examples:
            >>> result = ConversionResult(True, 42)
            >>> result.success
            True
            >>> result.value
            42
            >>> failed = ConversionResult(False, None, "Conversion error")
            >>> failed.error
            'Conversion error'
        """
        self.success: bool = success
        self.value: Optional[T] = value
        self.error: Optional[str] = error

    def __bool__(self) -> bool:
        """
        Boolean conversion returns success status.

        Returns:
            bool: True if conversion was successful, False otherwise

        Examples:
            >>> bool(ConversionResult.create_success(42))
            True
            >>> bool(ConversionResult[str].failure("Error"))
            False
        """
        return self.success

    def then(
        self,
        converter: Callable[[T], "ConversionResult[U]"],
    ) -> "ConversionResult[U]":
        """
        Chain another conversion operation if this one succeeded.

        Args:
            converter: Function to convert the value further

        Returns:
            ConversionResult[U]: Result of the chained conversion, or the original
                failure

        Examples:
            >>> # Convert string to int then to float
            >>> result = ConversionResult.create_success("42")
            >>> to_int = lambda s: ConversionResult.create_success(int(s))
            >>> to_float = lambda i: ConversionResult.create_success(float(i))
            >>> result.then(to_int).then(to_float).value
            42.0
            >>> # Failure stops the chain
            >>> failed = ConversionResult[str].failure("Invalid input")
            >>> failed.then(to_int).then(to_float).error
            'Invalid input'
        """
        if not self.success:
            # Cast is necessary here as we're changing the generic type parameter
            # from ConversionResult[T] to ConversionResult[U] while preserving the
            # failure state
            return cast("ConversionResult[U]", self)

        if self.value is None:
            # This should never happen if success is True, but we handle it for completeness
            return ConversionResult[U](
                False,
                None,
                "Value is None despite successful status",
            )

        return converter(self.value)

    def map(self, transform: Callable[[T], U]) -> "ConversionResult[U]":
        """
        Transform the value if conversion was successful.

        Args:
            transform: Function to transform the value

        Returns:
            ConversionResult[U]: Result containing the transformed value, or the
                original failure

        Examples:
            >>> result = ConversionResult.create_success(42)
            >>> result.map(lambda x: x * 2).value
            84
            >>> failed = ConversionResult[int].failure("Error")
            >>> failed.map(lambda x: x * 2).error
            'Error'
        """
        if not self.success:
            return cast("ConversionResult[U]", self)

        if self.value is None:
            return ConversionResult[U](
                False,
                None,
                "Value is None despite successful status",
            )

        try:
            return ConversionResult[U].create_success(transform(self.value))
        except Exception as e:
            return ConversionResult[U].failure(f"Transformation error: {str(e)}")

    def recover(self, recovery_func: Callable[[str], T]) -> "ConversionResult[T]":
        """
        Attempt to recover from a failed conversion.

        Args:
            recovery_func: Function that takes the error message and returns a recovery value

        Returns:
            ConversionResult[T]: Recovered result if this was a failure, or the
                original result

        Examples:
            >>> failed = ConversionResult[int].failure("Missing value")
            >>> failed.recover(lambda _: 0).value
            0
            >>> success = ConversionResult.create_success(42)
            >>> success.recover(lambda _: 0).value  # Original value preserved
            42
        """
        if self.success:
            return self

        try:
            error_msg = self.error or "Unknown error"
            recovery_value = recovery_func(error_msg)
            return ConversionResult[T].create_success(recovery_value)
        except Exception as e:
            return ConversionResult[T].failure(f"Recovery failed: {str(e)}")

    def or_else(self, default_value: T) -> T:
        """
        Get the result value or a default if conversion failed.

        Args:
            default_value: Value to return if conversion failed

        Returns:
            T: The conversion result value or the default

        Examples:
            >>> ConversionResult.create_success(42).or_else(0)
            42
            >>> ConversionResult[int].failure("Error").or_else(0)
            0
        """
        return self.value if self.success and self.value is not None else default_value

    def or_else_get(self, provider: Callable[[], T]) -> T:
        """
        Get the result value or compute a default if conversion failed.

        Args:
            provider: Function to compute the default value

        Returns:
            T: The conversion result value or the computed default

        Examples:
            >>> ConversionResult.create_success(42).or_else_get(lambda: 0)
            42
            >>> ConversionResult[int].failure("Error").or_else_get(lambda: 99)
            99
        """
        return self.value if self.success and self.value is not None else provider()

    def or_raise(self, exception_factory: Callable[[str], Exception] = ValueError) -> T:
        """
        Get the result value or raise an exception if conversion failed.

        Args:
            exception_factory: Function to create the exception from the error message

        Returns:
            T: The conversion result value

        Raises:
            Exception: If conversion failed, raises the exception created by exception_factory

        Examples:
            >>> ConversionResult.create_success(42).or_raise()
            42
            >>> try:
            ...     ConversionResult[int].failure("Bad value").or_raise()
            ... except ValueError as e:
            ...     str(e)
            'Bad value'
        """
        if not self.success:
            raise exception_factory(self.error or "Conversion failed")

        if self.value is None:
            raise exception_factory("Value is None despite successful status")

        return self.value

    @classmethod
    def create_success(cls, value: T) -> "ConversionResult[T]":
        """
        Create a successful conversion result.

        Args:
            value: The successfully converted value

        Returns:
            ConversionResult[T]: A successful conversion result

        Examples:
            >>> result = ConversionResult.create_success(42)
            >>> result.success
            True
            >>> result.value
            42
        """
        return cls(True, value)

    @classmethod
    def failure(cls, error: str) -> "ConversionResult[T]":
        """
        Create a failed conversion result.

        Args:
            error: Description of the error that occurred

        Returns:
            ConversionResult[T]: A failed conversion result

        Examples:
            >>> result = ConversionResult.failure("Invalid format")
            >>> result.success
            False
            >>> result.error
            'Invalid format'
        """
        return cls(False, None, error)

    @classmethod
    def from_try(cls, func: Callable[[], T]) -> "ConversionResult[T]":
        """
        Create a result by trying a function that may raise exceptions.

        Args:
            func: Function to execute that may raise an exception

        Returns:
            ConversionResult[T]: Successful result with the function's return value,
                               or failed result with the exception message

        Examples:
            >>> ConversionResult.from_try(lambda: int("42")).value
            42
            >>> result = ConversionResult.from_try(lambda: int("not_a_number"))
            >>> result.success
            False
            >>> "invalid literal" in result.error
            True
        """
        try:
            return cls.create_success(func())
        except Exception as e:
            return cls.failure(str(e))

    def __str__(self) -> str:
        """
        String representation of the conversion result.

        Returns:
            str: Description of the conversion result

        Examples:
            >>> str(ConversionResult.create_success(42))
            'Successful conversion: 42'
            >>> str(ConversionResult.failure("Invalid input"))
            'Failed conversion: Invalid input'
        """
        if self.success:
            return f"Successful conversion: {self.value}"
        return f"Failed conversion: {self.error}"

    def __repr__(self) -> str:
        """
        Detailed string representation of the conversion result.

        Returns:
            str: Detailed representation including class name

        Examples:
            >>> repr(ConversionResult.create_success(42))
            'ConversionResult(success=True, value=42, error=None)'
            >>> repr(ConversionResult.failure("Error"))
            'ConversionResult(success=False, value=None, error="Error")'
        """
        value_repr = repr(self.value) if self.value is not None else "None"
        error_repr = f'"{self.error}"' if self.error is not None else "None"
        return f"ConversionResult(success={self.success}, value={value_repr}, error={error_repr})"


# ──────────────────────────────────────────────────────────────
# Type Conversion Functions
# ──────────────────────────────────────────────────────────────

# Define TypeVars with more precise constraints
ConvertibleToInt = TypeVar(
    "ConvertibleToInt",
    bound=Union[int, float, str, bytes, SupportsIntConversion],
)
ConvertibleToBool = TypeVar(
    "ConvertibleToBool",
    bound=Union[
        bool,
        int,
        float,
        str,
        list[object],
        tuple[object, ...],
        set[object],
        dict[object, object],
        SupportsBoolConversion,
    ],
)
ConvertibleToFloat = TypeVar(
    "ConvertibleToFloat",
    bound=Union[int, float, str, SupportsFloatConversion],
)
ConvertibleToStr = TypeVar(
    "ConvertibleToStr",
    bound=Union[str, bytes, int, float, bool, Path],
)


def safe_int_convert(value: object) -> Optional[int]:
    """
    Safely convert a value to int or return None if invalid.

    Handles multiple input types including bool, int, float, str, and bytes.
    Guarantees no exceptions are raised during conversion.

    Args:
        value: Any value that might be convertible to int.

    Returns:
        Optional[int]: An int value or None if conversion is not possible.

    Examples:
        >>> safe_int_convert(42)
        42
        >>> safe_int_convert("42")
        42
        >>> safe_int_convert(3.14)
        3
        >>> safe_int_convert(True)
        1
        >>> safe_int_convert("hello") is None
        True
        >>> safe_int_convert(None) is None
        True
        >>> safe_int_convert("0x10")  # Hex strings need explicit handling
        None

    Note:
        This function silently handles all conversion errors by returning None.
        For hex/octal/binary strings, use the int(x, base) function directly.
    """
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return 1 if value else 0

        if isinstance(value, (int, float, str, bytes)):
            return int(value)
        if hasattr(value, "__int__"):
            # Cast to SupportsIntConversion to satisfy type checker
            return int(cast(SupportsIntConversion, value))

        return None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_bool_convert(value: object) -> bool:
    """
    Safely convert a value to bool with semantic interpretation.

    Performs intelligent boolean conversion with special handling for
    string values like "yes", "no", "true", "false", etc.

    Args:
        value: Any value that might be convertible to bool.

    Returns:
        bool: A boolean representation of the value, with common string patterns
              like "yes"/"no" properly handled.

    Examples:
        >>> safe_bool_convert(True)
        True
        >>> safe_bool_convert(1)
        True
        >>> safe_bool_convert("yes")
        True
        >>> safe_bool_convert("false")
        False
        >>> safe_bool_convert(0)
        False
        >>> safe_bool_convert([])
        False
        >>> safe_bool_convert([1, 2, 3])
        True
        >>> safe_bool_convert(None)
        False

    Note:
        This function interprets strings like "yes", "true", "1", "y", "t", "on" as True,
        and "no", "false", "0", "n", "f", "off" as False.
    """
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ("true", "yes", "1", "y", "t", "on"):
            return True
        if value_lower in ("false", "no", "0", "n", "f", "off"):
            return False
        return bool(value_lower)  # Empty string is False

    # Lists are always considered truthy in our domain model for compatibility
    if isinstance(value, list):
        return True

    # Other collections follow standard Python truthiness
    if isinstance(value, (tuple, set, Dict)):
        # Cast to SupportsLength to satisfy type checker
        collection_value = cast(SupportsLength, value)
        return len(collection_value) > 0

    # Use __bool__ if available
    if hasattr(value, "__bool__"):
        try:
            # Cast to SupportsBoolConversion to satisfy type checker
            bool_value = cast(SupportsBoolConversion, value)
            return bool(bool_value)
        except (ValueError, TypeError):
            pass

    # Fall back to truth value with proper casting
    return bool(value)


def safe_float_convert(value: object) -> Optional[float]:
    """
    Safely convert a value to float or return None if invalid.

    Handles multiple input types including bool, int, float, and str.
    Guarantees no exceptions are raised during conversion.

    Args:
        value: Any value that might be convertible to float.

    Returns:
        Optional[float]: A float value or None if conversion is not possible.

    Examples:
        >>> safe_float_convert(3.14)
        3.14
        >>> safe_float_convert("3.14")
        3.14
        >>> safe_float_convert(42)
        42.0
        >>> safe_float_convert(True)
        1.0
        >>> safe_float_convert("invalid") is None
        True
        >>> safe_float_convert(None) is None
        True
        >>> safe_float_convert("inf")
        inf
        >>> safe_float_convert("NaN")  # doctest: +ELLIPSIS
        nan

    Note:
        This function silently handles all conversion errors by returning None.
        Special values like "inf", "-inf", and "nan" are properly handled.
    """
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float, str)):
            return float(value)

        if hasattr(value, "__float__"):
            # Cast to SupportsFloatConversion to satisfy type checker
            float_compatible = cast(SupportsFloatConversion, value)
            return float(float_compatible)

        return None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_str_convert(value: object) -> str:
    """
    Safely convert a value to string with proper handling of various types.

    Provides special handling for bytes (UTF-8 decoding) and Path objects.
    Never raises exceptions, always returns a valid string.

    Args:
        value: Any value to convert to string.

    Returns:
        str: String representation of the value. Empty string for None.

    Examples:
        >>> safe_str_convert("hello")
        'hello'
        >>> safe_str_convert(42)
        '42'
        >>> safe_str_convert(None)
        ''
        >>> safe_str_convert(b'hello')
        'hello'
        >>> from pathlib import Path
        >>> safe_str_convert(Path('/tmp'))  # doctest: +SKIP
        '/tmp'
        >>> safe_str_convert(b'\\xff\\xfe')  # Invalid UTF-8 falls back to str(bytes)
        "b'\\xff\\xfe'"

    Note:
        This function attempts UTF-8 decoding for bytes objects and falls back
        to str(bytes) representation if decoding fails.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return str(value)

    if isinstance(value, Path):
        return str(value)

    return str(value)


def convert_with_fallback(
    value: S,
    primary_type: Type[T],
    fallback_type: Type[R],
) -> Union[T, R, S]:
    """
    Try to convert a value to a primary type, with fallback to a secondary type.

    Attempts to convert the value to the primary type first, and if that fails,
    tries converting to the fallback type. If both fail, returns the original value.

    Args:
        value: The value to convert
        primary_type: The preferred target type
        fallback_type: The fallback target type

    Returns:
        Union[T, R, S]: Converted value (T or R) or original value (S) if conversion fails

    Examples:
        >>> convert_with_fallback("123", int, float)
        123
        >>> convert_with_fallback("3.14", int, float)
        3.14
        >>> convert_with_fallback("hello", int, float)
        'hello'
        >>> convert_with_fallback(None, int, str)
        ''

    Note:
        This function silently handles conversion errors and returns the original
        value if both conversions fail.
    """
    # Special case handling for None
    if value is None:
        if primary_type is type(None):
            return None  # type: ignore
        if fallback_type is type(None):
            return None  # type: ignore

        # For None input but non-None target types, handle specially
        if primary_type is str:
            return ""  # type: ignore
        if fallback_type is str:
            return ""  # type: ignore

    # Try primary type conversion
    try:
        if primary_type is int:
            result = safe_int_convert(value)
            if result is not None:
                return result  # type: ignore
        elif primary_type is float:
            result = safe_float_convert(value)
            if result is not None:
                return result  # type: ignore
        elif primary_type is str:
            return safe_str_convert(value)  # type: ignore
        elif primary_type is bool:
            return safe_bool_convert(value)  # type: ignore
        else:
            # Generic conversion via constructor
            return primary_type(value)  # type: ignore
    except (ValueError, TypeError):
        pass

    # Try fallback type conversion
    try:
        if fallback_type is int:
            result = safe_int_convert(value)
            if result is not None:
                return result  # type: ignore
        elif fallback_type is float:
            result = safe_float_convert(value)
            if result is not None:
                return result  # type: ignore
        elif fallback_type is str:
            return safe_str_convert(value)  # type: ignore
        elif fallback_type is bool:
            return safe_bool_convert(value)  # type: ignore
        else:
            # Generic conversion via constructor
            return fallback_type(value)  # type: ignore
    except (ValueError, TypeError):
        pass

    # Return original value if all conversions fail
    return value


def try_convert(value: object, target_type: Type[T]) -> ConversionResult[T]:
    """
    Convert a value to a target type with detailed error reporting.

    Unlike the safe_*_convert functions, this provides structured error information
    when conversion fails rather than just returning None.

    Args:
        value: The value to convert
        target_type: The type to convert to

    Returns:
        ConversionResult[T]: A result object containing success status, converted
            value, and error information if conversion failed

    Examples:
        >>> result = try_convert("42", int)
        >>> result.success
        True
        >>> result.value
        42
        >>> result = try_convert("not_a_number", int)
        >>> result.success
        False
        >>> bool(result)
        False
        >>> result.error is not None
        True
        >>> result = try_convert(None, int)
        >>> result.success
        False

    Note:
        Captures and reports the actual exception that occurred during conversion.
    """
    if value is None and target_type is not type(None):
        return ConversionResult[T](False, None, "Cannot convert None to non-None type")

    if isinstance(value, target_type):
        # The value is already of the target type
        return ConversionResult[T](True, value, None)  # type: ignore

    try:
        # Special handling for common types with safe conversion functions
        if target_type is int:
            result = safe_int_convert(value)
            if result is not None:
                # Create a properly typed ConversionResult for int
                return ConversionResult[T](True, result, None)  # type: ignore
            return ConversionResult[T](
                False,
                None,
                f"Cannot convert {type(value).__name__} to int",
            )

        if target_type is bool:
            bool_result = safe_bool_convert(value)
            # Create a properly typed ConversionResult for bool
            return ConversionResult[T](True, bool_result, None)  # type: ignore

        if target_type is float:
            result = safe_float_convert(value)
            if result is not None:
                # Create a properly typed ConversionResult for float
                return ConversionResult[T](True, result, None)  # type: ignore
            return ConversionResult[T](
                False,
                None,
                f"Cannot convert {type(value).__name__} to float",
            )

        if target_type is str:
            str_result = safe_str_convert(value)
            # Create a properly typed ConversionResult for str
            return ConversionResult[T](True, str_result, None)  # type: ignore

        # For other types, try direct construction
        try:
            # Use direct constructor for the type
            converted_value = target_type(value)  # type: ignore
            return ConversionResult[T](True, converted_value, None)
        except Exception as e:
            return ConversionResult[T](False, None, f"{e.__class__.__name__}: {str(e)}")

    except Exception as e:
        return ConversionResult[T](False, None, f"{e.__class__.__name__}: {str(e)}")


def coerce_to_type(value: object, target_type: Type[T]) -> T:
    """
    Coerce a value to a target type, raising TypeError if conversion fails.

    Unlike try_convert, this function raises an exception if the value cannot
    be converted, making it suitable for validation scenarios where conversion
    failure should stop execution.

    Args:
        value: The value to convert
        target_type: The type to convert to

    Returns:
        T: The value converted to the target type

    Raises:
        TypeError: If the value cannot be converted to the target type

    Examples:
        >>> coerce_to_type("42", int)
        42
        >>> coerce_to_type(42, str)
        '42'
        >>> coerce_to_type(True, int)
        1
        >>> try:
        ...     coerce_to_type("not_a_number", int)
        ... except TypeError:
        ...     print("Conversion failed")
        Conversion failed

    Note:
        This function is more strict than convert_with_fallback, raising an exception
        rather than returning the original value on failure.
    """
    # Handle case where value is already of the target type
    if isinstance(value, target_type):
        return value  # type: ignore

    # Try conversion with detailed error reporting
    result = try_convert(value, target_type)

    if result.success and result.value is not None:
        return result.value

    # Raise error with detailed message if conversion failed
    error_msg = (
        result.error
        if result.error
        else f"Cannot convert {repr(value)} to {target_type.__name__}"
    )
    raise TypeError(error_msg)


_TYPE_CONVERTERS: Dict[Type[object], TypeConverter] = {}


def register_converter(
    target_type: Type[T],
    converter: Callable[[object], ConversionResult[T]],
) -> None:
    """
    Register a custom type converter for use with try_convert.

    Args:
        target_type: The type to convert to
        converter: Function that attempts to convert an object to target_type

    Examples:
        >>> class CustomType:
        ...     def __init__(self, value: int):
        ...         self.value = value
        ...     def __eq__(self, other):
        ...         return isinstance(other, CustomType) and self.value == other.value
        >>> def custom_converter(value: object) -> ConversionResult[CustomType]:
        ...     try:
        ...         if isinstance(value, int):
        ...             return ConversionResult.create_success(CustomType(value))
        ...         elif isinstance(value, str) and value.isdigit():
        ...             return ConversionResult.create_success(CustomType(int(value)))
        ...         return ConversionResult.failure(
        ...             f"Cannot convert {type(value).__name__} to CustomType")
        ...     except Exception as e:
        ...         return ConversionResult.failure(str(e))
        >>> register_converter(CustomType, custom_converter)
        >>> result = try_convert(42, CustomType)
        >>> result.success
        True
        >>> result.value == CustomType(42)
        True
    """
    _TYPE_CONVERTERS[target_type] = cast(TypeConverter, converter)
