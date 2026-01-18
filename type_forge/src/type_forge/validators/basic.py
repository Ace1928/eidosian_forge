from pathlib import Path
from typing import Dict, List, Optional, Sized, Type, TypeVar, cast

from ..core.base import BaseValidator, ValidationResult
from ..core.exceptions import TypeViolation, TypeViolationKind
from ..typing.protocols import (
    SupportsBoolConversion,
    SupportsFloatConversion,
    SupportsIntConversion,
)

T = TypeVar("T")


class BasicValidator(BaseValidator):
    """A class for basic data type validators."""

    @staticmethod
    def validate_string(value: object) -> str:
        """
        Validate that the value is a string.

        Args:
            value: The value to validate

        Returns:
            The validated string

        Raises:
            ValueError: If the value is not a string
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected a string, got {type(value).__name__}.")
        return value

    @staticmethod
    def validate_integer(value: object) -> int:
        """
        Validate that the value is an integer.

        Args:
            value: The value to validate

        Returns:
            The validated integer

        Raises:
            ValueError: If the value is not an integer
        """
        if not isinstance(value, int):
            raise ValueError(f"Expected an integer, got {type(value).__name__}.")
        return value

    @staticmethod
    def validate_float(value: object) -> float:
        """
        Validate that the value is a float.

        Args:
            value: The value to validate

        Returns:
            The validated float

        Raises:
            ValueError: If the value is not a float
        """
        if not isinstance(value, float):
            raise ValueError(f"Expected a float, got {type(value).__name__}.")
        return value

    @staticmethod
    def validate_boolean(value: object) -> bool:
        """
        Validate that the value is a boolean.

        Args:
            value: The value to validate

        Returns:
            The validated boolean

        Raises:
            ValueError: If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError(f"Expected a boolean, got {type(value).__name__}.")
        return value

    @staticmethod
    def validate_list(value: object, item_type: Type[T]) -> List[T]:
        """
        Validate that the value is a list of a specific item type.

        Args:
            value: The value to validate
            item_type: The expected type of items in the list

        Returns:
            The validated list

        Raises:
            ValueError: If the value is not a list or contains items of the wrong type
        """
        if not isinstance(value, list):
            raise ValueError(f"Expected a list, got {type(value).__name__}.")

        for item in value:
            if not isinstance(item, item_type):
                raise ValueError(
                    f"Expected list items of type {item_type.__name__}, got {type(item).__name__}."
                )

        return cast(List[T], value)

    @staticmethod
    def validate_dict(
        value: object, key_type: Type[T], value_type: Type[S]
    ) -> Dict[T, S]:
        """
        Validate that the value is a dictionary with specific key and value types.

        Args:
            value: The value to validate
            key_type: The expected type of keys in the dictionary
            value_type: The expected type of values in the dictionary

        Returns:
            The validated dictionary

        Raises:
            ValueError: If the value is not a dictionary or contains keys/values of the wrong type
        """
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dictionary, got {type(value).__name__}.")

        for key, val in value.items():
            if not isinstance(key, key_type):
                raise ValueError(
                    f"Expected key of type {key_type.__name__}, got {type(key).__name__}."
                )

            if not isinstance(val, value_type):
                raise ValueError(
                    f"Expected value of type {value_type.__name__}, got {type(val).__name__}."
                )

        return cast(Dict[T, S], value)

    @staticmethod
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

    @staticmethod
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
            return bool(len(cast(Sized, value)))

        # Use __bool__ if available
        if hasattr(value, "__bool__"):
            try:
                return bool(cast(SupportsBoolConversion, value))
            except (ValueError, TypeError):
                pass

        # Fall back to truth value
        return bool(value)

    @staticmethod
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

    @staticmethod
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

    def validate(self, value: object) -> bool:
        """
        Base validation method for the BasicValidator.

        Args:
            value: The value to validate

        Returns:
            True if validation succeeds, False otherwise
        """
        try:
            return value is not None
        except Exception:
            return False

    def validate_with_detail(self, value: object) -> ValidationResult[object]:
        """
        Validate with detailed information.

        Args:
            value: The value to validate

        Returns:
            ValidationResult with detailed information
        """
        try:
            is_valid = self.validate(value)
            return ValidationResult(
                valid=is_valid,
                violations=(
                    []
                    if is_valid
                    else [
                        TypeViolation(
                            path="$",
                            expected="valid value",
                            found=str(type(value).__name__),
                            kind=TypeViolationKind.INVALID_VALUE,
                        )
                    ]
                ),
                converted_value=value if is_valid else None,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                violations=[
                    TypeViolation(
                        path="$",
                        expected="valid value",
                        found=str(e),
                        kind=TypeViolationKind.INVALID_VALUE,
                    )
                ],
            )
