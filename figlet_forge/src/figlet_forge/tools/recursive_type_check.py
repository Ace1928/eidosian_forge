"""
Recursive Type System for Figlet Forge

Implements a comprehensive type validation framework with:
- Recursive type definitions for complex nested structures
- Runtime type verification with precise path reporting
- Type conversion utilities with strict safety guarantees
- Integration with static type checkers while preserving runtime flexibility
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

# Configure logger for this module
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Type Protocol Definitions
# ──────────────────────────────────────────────────────────────


class SupportsIntConversion(Protocol):
    """Protocol for types that can be converted to int."""

    def __int__(self) -> int: ...


class SupportsBoolConversion(Protocol):
    """Protocol for types that can be converted to bool."""

    def __bool__(self) -> bool: ...


class SupportsStrConversion(Protocol):
    """Protocol for types that can be converted to str."""

    def __str__(self) -> str: ...


class SupportsFloatConversion(Protocol):
    """Protocol for types that can be converted to float."""

    def __float__(self) -> float: ...


# ──────────────────────────────────────────────────────────────
# Recursive Type Definitions with Maximum Precision
# ──────────────────────────────────────────────────────────────

T = TypeVar("T", bound=object)
R = TypeVar("R", bound=object)
K = TypeVar("K", str, int)  # Key types for mappings
S = TypeVar("S", bound=object)  # Additional type var for schemas

# Define atomic types
AtomicValueT = Union[str, int, bool, float, None]

# Define collection types recursively with proper forward references
DetailValueT = Union[
    AtomicValueT,
    List[str],
    Sequence["DetailValueT"],  # More generic than just List
    Dict[str, "DetailValueT"],  # Dictionary mapping
    Mapping[str, "DetailValueT"],  # More generic mapping
]

# Define explicitly convertible types with semantic meaning
ConvertibleToInt = Union[int, float, str, bytes, SupportsIntConversion]
ConvertibleToBool = Union[
    bool, int, float, str, SupportsIntConversion, SupportsBoolConversion
]
ConvertibleToFloat = Union[int, float, str, SupportsFloatConversion]
ConvertibleToStr = Union[str, int, float, bool, bytes, Path, SupportsStrConversion]

# Simple types for direct use in kwargs
SimpleValueT = Union[str, int, bool, float, List[str], None]

# Kwargs specialized type with narrower constraints for safer validation
KwargValueT = Union[
    SimpleValueT,
    Dict[str, SimpleValueT],
    List[SimpleValueT],
]

KwargsT = Dict[str, KwargValueT]

# Dict parameter type using Mapping for covariance
DictParamT = Optional[Mapping[str, DetailValueT]]

# Type aliases for common validation patterns - using proper recursive structures
# Define SchemaValueT recursively to handle nested schemas
SchemaValueT = Union[
    Type[object],
    Tuple[Type[object], ...],
    Sequence[Type[object]],
    Mapping[str, "SchemaValueT"],  # Allow recursive schema definitions
]

# Use Mapping for covariance rather than Dict (which is invariant)
DictSchemaT = Mapping[str, SchemaValueT]

# Schema types for recursive validation with proper variance
SchemaTypeT = Union[
    DictSchemaT,
    Sequence[SchemaValueT],  # Using Sequence instead of List for covariance
    Type[object],
    Tuple[Type[object], ...],
]


# ──────────────────────────────────────────────────────────────
# Type Verification Result Classes
# ──────────────────────────────────────────────────────────────


class TypeViolationKind(Enum):
    """Enumeration of possible type violation kinds."""

    WRONG_TYPE = "wrong_type"
    MISSING_KEY = "missing_key"
    INVALID_VALUE = "invalid_value"
    SCHEMA_MISMATCH = "schema_mismatch"
    CONVERSION_ERROR = "conversion_error"


@dataclass(frozen=True)
class TypeViolation:
    """Immutable record of a type violation with path tracking."""

    path: str
    expected: str
    found: str
    kind: TypeViolationKind

    def __str__(self) -> str:
        """Generate human-readable representation of the violation."""
        return (
            f"At path '{self.path}': Expected {self.expected}, "
            f"found {self.found} ({self.kind.value})"
        )


@dataclass
class ValidationResult(Generic[T]):
    """Result of type validation with possible conversion."""

    valid: bool
    violations: List[TypeViolation]
    converted_value: Optional[T] = None

    def __bool__(self) -> bool:
        """
        Boolean conversion returns validation status.

        Returns:
            True if validation passed, False otherwise
        """
        return self.valid

    def merge(self, other: "ValidationResult[object]") -> "ValidationResult[T]":
        """
        Merge another validation result into this one.

        This maintains the type of the original result while incorporating
        the validation status and violations from another result.

        Args:
            other: Another validation result to merge with this one

        Returns:
            Self with merged validation status
        """
        if not other.valid:
            self.valid = False
        self.violations.extend(other.violations)
        return self


# ──────────────────────────────────────────────────────────────
# Core Type Conversion Functions with Enhanced Safety
# ──────────────────────────────────────────────────────────────


def safe_int_convert(value: object) -> Optional[int]:
    """
    Safely convert a value to int or return None if invalid.

    Args:
        value: Any value that might be convertible to int

    Returns:
        An int value or None if conversion is not possible
    """
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return 1 if value else 0

        if isinstance(value, (int, float, str, bytes)):
            return int(value)

        if hasattr(value, "__int__"):
            return int(cast(SupportsIntConversion, value))

        return None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_bool_convert(value: object) -> bool:
    """
    Safely convert a value to bool with semantic interpretation.

    Args:
        value: Any value that might be convertible to bool

    Returns:
        A bool representation of the value, with common string patterns
        like "yes"/"no" properly handled
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
    if isinstance(value, (tuple, set)):
        return len(cast(Sized, value)) > 0

    # Use __bool__ if available
    if hasattr(value, "__bool__"):
        try:
            return bool(cast(SupportsBoolConversion, value))
        except (ValueError, TypeError):
            pass

    # Fall back to truth value
    return bool(value)


def safe_float_convert(value: object) -> Optional[float]:
    """
    Safely convert a value to float or return None if invalid.

    Args:
        value: Any value that might be convertible to float

    Returns:
        A float value or None if conversion is not possible
    """
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float, str)):
            return float(value)

        if hasattr(value, "__float__"):
            return float(cast(SupportsFloatConversion, value))

        return None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_str_convert(value: object) -> str:
    """
    Safely convert a value to string with proper handling of various types.

    Args:
        value: Any value to convert to string

    Returns:
        String representation of the value
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


# ──────────────────────────────────────────────────────────────
# Recursive Type Checking Core
# ──────────────────────────────────────────────────────────────


class TypeChecker:
    """
    Comprehensive recursive type checking system with path tracking and error reporting.

    Features:
    - Type validation for complex nested structures
    - Recursive path tracking for precise error location
    - Optional value conversion with validation
    - Support for both static type checking and runtime verification
    """

    @staticmethod
    def validate_type(
        value: object,
        expected_type: Union[Type[T], Sequence[Type[object]]],
        path: str = "$",
        convert: bool = False,
    ) -> ValidationResult[T]:
        """
        Validate that a value matches the expected type recursively.

        Args:
            value: The value to check
            expected_type: Type or types to check against
            path: Current path in the validation (for error reporting)
            convert: Whether to attempt conversion to the expected type

        Returns:
            ValidationResult with validation status and details
        """
        # Normalize expected_type to a tuple
        if isinstance(expected_type, type):
            normalized_types: Tuple[Type[object], ...] = (expected_type,)
        else:
            # Convert sequence of types to tuple for consistent handling
            normalized_types = tuple(t for t in expected_type)

        # Handle None specially
        if value is None:
            if type(None) in normalized_types:
                return ValidationResult(True, [], None)

            type_names = ", ".join(
                t.__name__ if hasattr(t, "__name__") else str(t)
                for t in normalized_types
            )

            return ValidationResult(
                False,
                [
                    TypeViolation(
                        path=path,
                        expected=type_names,
                        found="None",
                        kind=TypeViolationKind.WRONG_TYPE,
                    )
                ],
                None,
            )

        # Check if value matches any of the expected types
        if isinstance(value, normalized_types):
            return ValidationResult(True, [], cast(T, value))

        # Attempt conversion if requested
        if convert:
            for typ in normalized_types:
                # Skip None type in conversion attempts
                if typ is type(None):
                    continue

                converted = TypeChecker._try_convert(value, typ)
                if converted is not None:
                    return ValidationResult(True, [], cast(T, converted))

        # Value doesn't match and couldn't be converted
        type_names = ", ".join(
            t.__name__ if hasattr(t, "__name__") else str(t) for t in normalized_types
        )

        return ValidationResult(
            False,
            [
                TypeViolation(
                    path=path,
                    expected=type_names,
                    found=type(value).__name__,
                    kind=TypeViolationKind.WRONG_TYPE,
                )
            ],
            None,
        )

    @staticmethod
    def validate_dict(
        value: object,
        schema: DictSchemaT,
        path: str = "$",
        convert: bool = False,
        require_all_keys: bool = True,
    ) -> ValidationResult[Dict[str, object]]:
        """
        Validate that a dictionary conforms to a schema.

        Args:
            value: Dictionary to validate
            schema: Schema defining expected types for keys
            path: Current path for error reporting
            convert: Whether to attempt type conversion
            require_all_keys: Whether all schema keys must be present

        Returns:
            ValidationResult with validation status and converted dict if applicable
        """
        if not isinstance(value, dict):
            return ValidationResult(
                False,
                [
                    TypeViolation(
                        path=path,
                        expected="dict",
                        found=type(value).__name__,
                        kind=TypeViolationKind.WRONG_TYPE,
                    )
                ],
                None,
            )

        # We can safely cast now that we've verified it's a dict
        dict_value = cast(Dict[str, object], value)

        result_dict: Dict[str, object] = {}
        result: ValidationResult[Dict[str, object]] = ValidationResult(
            True, [], result_dict
        )

        # Check for required keys
        if require_all_keys:
            missing_keys = [k for k in schema if k not in dict_value]
            if missing_keys:
                for key in missing_keys:
                    expected_type = schema[key]
                    expected_str = TypeChecker._get_type_name(expected_type)

                    result.violations.append(
                        TypeViolation(
                            path=f"{path}.{key}",
                            expected=expected_str,
                            found="missing",
                            kind=TypeViolationKind.MISSING_KEY,
                        )
                    )
                result.valid = False

        # Validate each present key
        for key, expected_type in schema.items():
            if key in dict_value:
                key_path = f"{path}.{key}"
                key_result = TypeChecker.validate_recursive(
                    dict_value[key], cast(SchemaTypeT, expected_type), key_path, convert
                )

                if key_result.valid and key_result.converted_value is not None:
                    result_dict[key] = key_result.converted_value
                else:
                    result_dict[key] = dict_value[key]

                result.merge(key_result)

        return result

    @staticmethod
    def _get_type_name(typ: SchemaValueT) -> str:
        """
        Get a human-readable name for a type or type collection.

        Args:
            typ: A type, tuple of types, or list/sequence of types

        Returns:
            A string representation of the type(s)
        """
        if isinstance(typ, type):
            return typ.__name__
        # Handle tuples of types (Union-like)
        elif isinstance(typ, tuple) and all(isinstance(t, type) for t in typ):  # type: ignore # Necessary to ensure that the casting isn't picked up wrong
            return " | ".join(t.__name__ for t in typ)
        # Handle lists/sequences of types or values
        elif isinstance(typ, (list, tuple, Sequence)) and not isinstance(
            typ, (str, bytes)
        ):
            if len(typ) > 0:
                if isinstance(typ[0], type):  # type: ignore # Necessary to ensure that the casting isn't picked up wrong
                    return f"List[{typ[0].__name__}]"
                else:
                    return "List[Dict]"  # For list of schema dicts
            else:
                return "List[Any]"
        # Handle dictionaries (structural types)
        elif isinstance(typ, dict):
            return "Dict[str, Any]"
        else:
            return str(typ)

    @staticmethod
    def validate_recursive(
        value: object,
        schema: SchemaTypeT,
        path: str = "$",
        convert: bool = False,
    ) -> ValidationResult[object]:
        """
        Recursively validate a value against a schema of arbitrary complexity.

        Args:
            value: Value to validate
            schema: Schema to validate against (dict, list/sequence, or type)
            path: Current path for error reporting
            convert: Whether to attempt type conversion

        Returns:
            ValidationResult with validation status and details
        """
        # Validate schema type
        if not isinstance(schema, (dict, list, tuple, type)) and not (
            isinstance(schema, tuple) and all(isinstance(t, type) for t in schema)
        ):
            raise TypeError(
                f"Invalid schema type: {type(schema).__name__}. "
                f"Expected dict, list, tuple of types, or type."
            )

        # Handle tuple of types (Union-like behavior) - check this case first
        if isinstance(schema, tuple) and all(isinstance(t, type) for t in schema):
            return TypeChecker.validate_type(
                value, cast(Tuple[Type[object], ...], schema), path, convert
            )

        # Handle dict schema
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                return ValidationResult(
                    False,
                    [
                        TypeViolation(
                            path=path,
                            expected="dict",
                            found=type(value).__name__,
                            kind=TypeViolationKind.WRONG_TYPE,
                        )
                    ],
                    None,
                )
            # Cast the return value to ensure type compatibility
            dict_result = TypeChecker.validate_dict(
                cast(Dict[str, object], value), cast(DictSchemaT, schema), path, convert
            )
            return cast(ValidationResult[object], dict_result)

        # Handle list/sequence schema
        elif (
            isinstance(schema, (list, tuple, Sequence))
            and len(schema) > 0
            and not isinstance(schema, (str, bytes))
        ):
            if not isinstance(value, (list, tuple)):
                return ValidationResult(
                    False,
                    [
                        TypeViolation(
                            path=path,
                            expected="list or tuple",
                            found=type(value).__name__,
                            kind=TypeViolationKind.WRONG_TYPE,
                        )
                    ],
                    None,
                )

            result: ValidationResult[List[object]] = ValidationResult(True, [], [])
            element_schema = schema[0]
            result_list: List[object] = []

            # Properly type the value as a sequence for iteration
            sequence_value = cast(Sequence[object], value)

            for i, item in enumerate(sequence_value):
                item_path = f"{path}[{i}]"
                item_result = TypeChecker.validate_recursive(
                    item, cast(SchemaTypeT, element_schema), item_path, convert
                )
                if not item_result.valid:
                    result.valid = False
                    result.violations.extend(item_result.violations)

                # Always add the item (original or converted) to the result list
                result_list.append(
                    item_result.converted_value
                    if item_result.valid and item_result.converted_value is not None
                    else item
                )

            # Set the converted value if validation succeeded
            if result.valid:
                result.converted_value = result_list

            return cast(ValidationResult[object], result)

        # Handle simple type validation
        elif isinstance(schema, type):
            type_result = TypeChecker.validate_type(value, schema, path, convert)
            return cast(ValidationResult[object], type_result)  # type: ignore # Necessary to ensure that the casting isn't picked up wrong

        # Default case for invalid schema
        else:
            return ValidationResult(
                False,
                [
                    TypeViolation(
                        path=path,
                        expected="valid schema",
                        found=f"invalid schema: {type(schema).__name__}",
                        kind=TypeViolationKind.SCHEMA_MISMATCH,
                    )
                ],
                None,
            )

    @staticmethod
    def _try_convert(value: object, target_type: Type[T]) -> Optional[T]:
        """
        Try to convert a value to the target type.

        Args:
            value: Value to convert
            target_type: Type to convert to

        Returns:
            Converted value or None if conversion failed
        """
        try:
            # Handle built-in types
            if target_type is int:
                result = safe_int_convert(value)
                return cast(T, result) if result is not None else None

            if target_type is bool:
                return cast(T, safe_bool_convert(value))

            if target_type is float:
                result = safe_float_convert(value)
                return cast(T, result) if result is not None else None

            if target_type is str:
                return cast(T, safe_str_convert(value))

            # For other types, attempt direct construction
            if not isinstance(value, target_type):
                try:
                    # Use explicit constructor
                    return target_type(value)  # type: ignore
                except (ValueError, TypeError):
                    pass

            return None
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────
# Public API Functions
# ──────────────────────────────────────────────────────────────


@overload
def validate_type(value: object, expected_type: Type[T]) -> ValidationResult[T]: ...


@overload
def validate_type(
    value: object, expected_type: Sequence[Type[object]]
) -> ValidationResult[object]: ...


def validate_type(
    value: object,
    expected_type: Union[Type[T], Sequence[Type[object]]],
) -> Union[ValidationResult[T], ValidationResult[object]]:
    """
    Validate that a value matches the expected type.

    Args:
        value: Value to validate
        expected_type: Type or types to check against

    Returns:
        ValidationResult with validation details
    """
    return TypeChecker.validate_type(value, expected_type)


def validate_and_convert(
    value: object, target_type: Type[T], path: str = "$"
) -> ValidationResult[T]:
    """
    Validate and convert a value to the target type.

    Args:
        value: Value to validate and convert
        target_type: Type to convert to
        path: Path for error reporting

    Returns:
        ValidationResult with conversion result
    """
    return TypeChecker.validate_type(value, target_type, path, convert=True)


def validate_dict_schema(
    data: object,
    schema: DictSchemaT,
    convert: bool = False,
    require_all_keys: bool = False,  # Changed default to match test expectations
) -> ValidationResult[Dict[str, object]]:
    """
    Validate that a dictionary conforms to a schema.

    Args:
        data: Dictionary to validate
        schema: Schema defining expected types for keys
        convert: Whether to attempt type conversion
        require_all_keys: Whether all schema keys must be present in data

    Returns:
        ValidationResult with validation status and details
    """
    return TypeChecker.validate_dict(
        data, schema, path="$", convert=convert, require_all_keys=require_all_keys
    )


@overload
def check_type(value: object, expected_type: Type[T]) -> bool: ...


@overload
def check_type(value: object, expected_type: Tuple[Type[object], ...]) -> bool: ...


def check_type(
    value: object, expected_type: Union[Type[object], Tuple[Type[object], ...]]
) -> bool:
    """
    Simple type check without detailed reporting.

    Args:
        value: Value to check
        expected_type: Type or types to check against

    Returns:
        True if value matches expected type, False otherwise
    """
    return TypeChecker.validate_type(value, expected_type).valid


@overload
def assert_type(
    value: object, expected_type: Type[T], message: Optional[str] = None
) -> T: ...


@overload
def assert_type(
    value: object,
    expected_type: Tuple[Type[object], ...],
    message: Optional[str] = None,
) -> object: ...


def assert_type(
    value: object,
    expected_type: Union[Type[T], Tuple[Type[object], ...]],
    message: Optional[str] = None,
) -> Union[T, object]:
    """
    Assert that a value has the expected type.

    Args:
        value: Value to check
        expected_type: Type or types to check against
        message: Custom error message

    Returns:
        The original value if it has the expected type

    Raises:
        TypeError: If value doesn't match the expected type
    """
    result = TypeChecker.validate_type(value, expected_type)
    if not result.valid:
        error_msg = message if message is not None else ""
        if not error_msg:
            violations = "\n".join(str(v) for v in result.violations)
            error_msg = f"Type assertion failed: {violations}"
        raise TypeError(error_msg)
    return value  # Already checked by validate_type
