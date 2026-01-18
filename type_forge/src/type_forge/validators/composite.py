from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from ..core.base import BaseValidator, ValidationResult
from ..core.exceptions import TypeViolation, TypeViolationKind

T = TypeVar("T")


class CompositeValidator(BaseValidator, Generic[T]):
    """A validator that combines multiple validators."""

    def __init__(self, validators: List[Callable[[object], bool]]) -> None:
        """
        Initialize a composite validator with a list of validator functions.

        Args:
            validators: List of validator functions that take a value and return a boolean
        """
        self.validators = validators

    def validate(self, value: object) -> bool:
        """
        Validate a value using all validators in the composite.

        Args:
            value: The value to validate

        Returns:
            True if all validators return True, False otherwise
        """
        return all(validator(value) for validator in self.validators)

    def __call__(self, value: object) -> bool:
        """
        Make the validator callable directly.

        Args:
            value: The value to validate

        Returns:
            Result of validate(value)
        """
        return self.validate(value)

    def add_validator(self, validator: Callable[[object], bool]) -> None:
        """
        Add a new validator to the composite.

        Args:
            validator: A function that takes a value and returns a boolean
        """
        self.validators.append(validator)

    def validate_with_details(
        self, value: object, path: str = "$"
    ) -> ValidationResult[T]:
        """
        Validate with detailed result information.

        Args:
            value: The value to validate
            path: The path to report in case of violations

        Returns:
            A ValidationResult with detailed information
        """
        violations: List[TypeViolation] = []
        valid = True

        for i, validator in enumerate(self.validators):
            if not validator(value):
                valid = False
                violations.append(
                    TypeViolation(
                        path=f"{path}.validator[{i}]",
                        expected="valid value",
                        found=str(type(value).__name__),
                        kind=TypeViolationKind.INVALID_VALUE,
                    )
                )

        return ValidationResult(
            valid=valid,
            violations=violations,
            converted_value=cast(T, value) if valid else None,
        )

    @staticmethod
    def from_validators(
        validators: Sequence[BaseValidator],
    ) -> "CompositeValidator[object]":
        """
        Create a CompositeValidator from a sequence of BaseValidator instances.

        Args:
            validators: A sequence of BaseValidator instances

        Returns:
            A new CompositeValidator that aggregates all the validators
        """
        validator_funcs = [
            lambda v, validator=v: validator.validate(v) for v in validators
        ]
        return CompositeValidator(validator_funcs)


def is_not_empty(value: str) -> bool:
    """Check if a string is not empty."""
    return bool(value)


def is_positive(value: int) -> bool:
    """Check if an integer is positive."""
    return value > 0


def is_in_range(value: int, min_value: int, max_value: int) -> bool:
    """Check if an integer is within a specified range."""
    return min_value <= value <= max_value


def is_valid_length(
    value: Collection[Any],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> bool:
    """
    Check if a collection has a valid length.

    Args:
        value: The collection to check
        min_length: Minimum allowed length (inclusive), or None for no minimum
        max_length: Maximum allowed length (inclusive), or None for no maximum

    Returns:
        True if the length is valid, False otherwise
    """
    length = len(value)
    if min_length is not None and length < min_length:
        return False
    if max_length is not None and length > max_length:
        return False
    return True
