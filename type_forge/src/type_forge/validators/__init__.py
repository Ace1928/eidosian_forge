# Contents of /type-forge/type-forge/src/type_forge/validators/__init__.py

"""Validators module for type checking and validation.

This module provides a comprehensive set of validators for data verification
and type checking. It centralizes the validation functionality through a
structured interface of basic validators, composite validators, and validator
factories.

Each validator follows the BaseValidator interface pattern, enabling consistent
composition, chaining, and extension with uniform error handling.

Available Validators:
    - Basic Validators: Core validators for primitive types and simple structures
    - Composite Validators: Combinators for creating complex validation rules
    - Validator Factory: Dynamic validator creation and registration

Example:
    >>> from type_forge.validators import BasicValidator, CompositeValidator
    >>> integer_validator = BasicValidator()
    >>> result = integer_validator.validate(42)

Attributes:
    BaseValidator: Abstract base class for all validators
    BasicValidator: Simple validators for primitive types
    CompositeValidator: Combines multiple validators
    ValidatorFactory: Creates validators dynamically
"""

from typing import List

# Import core validator classes from typing.validation
from type_forge.typing.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    is_callable,
    is_collection,
    is_instance_of_any,
    is_non_empty_string,
    is_numeric,
    is_protocol_instance,
    is_subclass_safe,
    is_valid_identifier,
    version,
)

# Import basic validators
from type_forge.validators.basic import (
    BasicValidator,
    SupportsBoolConversion,
    SupportsFloatConversion,
    SupportsIntConversion,
    T,
)

# Import composite validators
from type_forge.validators.composite import (
    CompositeValidator,
    is_in_range,
    is_not_empty,
    is_positive,
    is_valid_length,
)

# Import validator factory
from type_forge.validators.factory import V, ValidatorFactory

# Define public exports explicitly
__all__: List[str] = [
    # Core validation components
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "is_callable",
    "is_collection",
    "is_instance_of_any",
    "is_non_empty_string",
    "is_numeric",
    "is_protocol_instance",
    "is_subclass_safe",
    "is_valid_identifier",
    "version",
    # Basic validators
    "BasicValidator",
    "SupportsFloatConversion",
    "SupportsIntConversion",
    "SupportsBoolConversion",
    "T",
    # Composite validators
    "CompositeValidator",
    "is_in_range",
    "is_not_empty",
    "is_positive",
    "is_valid_length",
    # Factory components
    "V",
    "ValidatorFactory",
]
