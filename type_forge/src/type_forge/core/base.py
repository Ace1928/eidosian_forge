"""Core validation framework with precise type guarantees and recursive refinement.

This module provides the fundamental building blocks for type validation with
detailed reporting capabilities that maintain type safety through generics.

The validation framework implements recursive principles, allowing validators
to be composed and results to be merged while maintaining type integrity
throughout the validation pipeline. Through this composition pattern,
complex validation logic emerges from simple, atomic validators.

Key components:
    ValidationResult: Generic container preserving type information through validation
    BaseValidator: Abstract interface for implementing validation logic
    TypeForgeBase: Composition mechanism for creating validator chains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, List, Optional

# Resolve circular import by importing at the top
from ..core.exceptions import TypeViolation, TypeViolationKind
from ..typing.definitions import S, T


@dataclass
class ValidationResult(Generic[T]):
    """Result of type validation with possible conversion.

    This class encapsulates the outcome of a validation operation, including
    whether validation passed, any violations that occurred, and an optional
    converted value that maintains its type through generic constraints.

    The ValidationResult maintains type safety through covariant generics,
    ensuring that type information flows correctly through validation chains
    and transformations. It acts as both a container for validation status
    and a monad-like structure that can be composed and transformed while
    preserving the validation context.

    Attributes:
        valid (bool): Boolean indicating if validation succeeded
        violations (List[TypeViolation]): List of specific type violations encountered
        converted_value (Optional[T]): Optional transformed value that maintains
            its type through generics

    Examples:
        >>> result = ValidationResult[int](valid=True, converted_value=42)
        >>> bool(result)
        True
        >>> result.with_converted_value("string")
        ValidationResult(valid=True, violations=[], converted_value='string')
    """

    valid: bool
    violations: List[TypeViolation] = field(default_factory=list)
    converted_value: Optional[T] = None

    def __bool__(self) -> bool:
        """Boolean conversion returns validation status.

        Allows ValidationResult objects to be used directly in boolean contexts
        for streamlined conditional logic, implementing the "errors as values"
        pattern in an elegant, composable way.

        Returns:
            bool: True if validation passed, False otherwise
        """
        return self.valid

    def merge(self, other: ValidationResult[object]) -> ValidationResult[T]:
        """Merge another validation result into this one.

        This maintains the type of the original result while incorporating
        the validation status and violations from another result. The converted
        value remains that of the original result, preserving type integrity.

        This method enables recursive composition of validation results,
        allowing complex validation logic to be built from simpler components
        while maintaining full type safety.

        Args:
            other (ValidationResult[object]): Another validation result to merge with this one

        Returns:
            ValidationResult[T]: Self with merged validation status and violations
        """
        if not other.valid:
            self.valid = False
        self.violations.extend(other.violations)
        return self

    def with_converted_value(self, value: S) -> ValidationResult[S]:
        """Create a new ValidationResult with the same validation status but a different value.

        This method preserves the validation state while transforming the result
        to contain a new value of potentially different type, properly maintaining
        type safety through generics. This enables validation pipelines that
        transform data while maintaining error context.

        The implementation follows the principle of "Pure Core, Effects at Edges"
        by returning a new instance rather than modifying the current one,
        maintaining immutability for better compositional properties.

        Args:
            value (S): The converted value to set

        Returns:
            ValidationResult[S]: A new ValidationResult with the same validation status but different value
        """
        return ValidationResult(
            valid=self.valid, violations=self.violations.copy(), converted_value=value
        )


class BaseValidator:
    """Base class for all validators in the type_forge framework.

    Provides the fundamental validation interface that all validators must implement,
    with support for both simple boolean validation and detailed validation results.

    Validators form the core of the type forge validation process, each implementing
    specific validation logic while adhering to a common interface that enables
    composition and chaining. This follows the "Composition Over Inheritance"
    principle from Eidosian design.

    The class follows the Template Method pattern, providing a default implementation
    of validate_with_detail that builds upon the abstract validate method
    that subclasses must implement.
    """

    def validate(self, value: object) -> bool:
        """Validate the given value.

        This is the simplified validation interface that returns only a boolean result.
        Subclasses must implement this method to define specific validation logic.

        Args:
            value (object): The value to validate

        Returns:
            bool: True if the value is valid, False otherwise

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        _ = value  # Mark the parameter as used
        raise NotImplementedError("Subclasses must implement this method.")

    def validate_with_detail(self, value: object) -> ValidationResult[object]:
        """Validate the given value with detailed results.

        Provides comprehensive validation information including specific violations
        and an optionally converted value. This implementation relies on the simple
        validate method, but subclasses may override for more detailed reporting.

        This method demonstrates the recursive refinement principle by building
        detailed validation upon the simpler boolean validation. It also implements
        the "Errors as Values" principle by encapsulating failures in the return
        type rather than through exceptions.

        Args:
            value (object): The value to validate

        Returns:
            ValidationResult[object]: Detailed validation results with type preservation

        Raises:
            No direct exceptions, but captures exceptions from validate() and
            converts them to ValidationResult with appropriate violations
        """
        try:
            is_valid: bool = self.validate(value)
            return ValidationResult[object](
                valid=is_valid,
                violations=[],
                converted_value=value if is_valid else None,
            )
        except Exception as e:
            return ValidationResult[object](
                valid=False,
                violations=[
                    TypeViolation(
                        path="$",
                        expected=self.__class__.__name__,
                        found=str(e),
                        kind=TypeViolationKind.WRONG_TYPE,
                    )
                ],
                converted_value=None,
            )


class TypeForgeBase:
    """Base class for the type forging process.

    Orchestrates validation through multiple validators, providing both
    simple boolean validation and detailed validation results with proper
    type preservation.

    The TypeForgeBase implements the composition pattern, allowing multiple
    validators to be combined while maintaining a consistent interface
    and preserving type information throughout the validation process.
    This embodies the Eidosian principle of "Fractal Coherence" where
    complex validation logic emerges from simpler components in a
    consistent manner.

    Attributes:
        validators (List[BaseValidator]): List of validators to apply during validation
    """

    def __init__(self) -> None:
        """Initialize with an empty validators list.

        Creates a new TypeForgeBase instance with no validators.
        Validators must be added using the add_validator method.
        This follows the "Data Before Behavior" principle by establishing
        the core data structure before defining operations on it.
        """
        self.validators: List[BaseValidator] = []

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the type forging process.

        Appends a validator to the internal list of validators that will
        be applied during validation operations. This method implements
        the builder pattern for constructing validation chains.

        Args:
            validator (BaseValidator): An instance of a validator

        Raises:
            TypeError: If validator is not an instance of BaseValidator
        """
        if not validator:
            raise TypeError("Validator must be an instance of BaseValidator.")
        self.validators.append(validator)

    def validate(self, value: object) -> bool:
        """Validate a value against all registered validators.

        Performs sequential validation using all registered validators,
        implementing the fail-fast principle by short-circuiting on the first failure
        for efficient validation.

        This method provides a simplified boolean interface to the validation
        process, while maintaining full compatibility with the type registration
        and validation system used throughout TypeForge.

        Args:
            value (object): The value to validate against the validators
                in this TypeForgeBase instance

        Returns:
            bool: True if the value passes all validators, False otherwise

        Examples:
            >>> validator = TypeForgeBase()
            >>> validator.add_validator(IntegerValidator())
            >>> validator.validate(42)
            True
            >>> validator.validate("not an integer")
            False
        """
        if not self.validators:
            return True  # If no validators are registered, validation passes

        return all(validator.validate(value) for validator in self.validators)

    def validate_with_detail(self, value: object) -> ValidationResult[object]:
        """Validate with detailed results from all validators.

        Aggregates validation results from all validators, maintaining a comprehensive
        record of any violations that occur while preserving type safety.

        Unlike the simple validate method, this continues to run all validators
        even after failures to collect complete violation information.
        This implementation balances efficiency with completeness, providing
        full failure details for better debugging and error reporting.

        Args:
            value (object): The value to validate

        Returns:
            ValidationResult[object]: Aggregated validation details with type preservation
        """
        result: ValidationResult[object] = ValidationResult[object](
            valid=True, violations=[], converted_value=value
        )

        for validator in self.validators:
            validator_result: ValidationResult[object] = validator.validate_with_detail(
                value
            )
            result.merge(validator_result)

        return result
