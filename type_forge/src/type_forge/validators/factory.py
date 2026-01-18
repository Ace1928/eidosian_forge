from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from ..core.base import ValidationResult
from ..core.exceptions import TypeViolation, TypeViolationKind
from ..typing.hints import DictSchemaT, SchemaTypeT, T
from .basic import BasicValidator
from .composite import CompositeValidator

# Type variable for generic validator functions
V = TypeVar("V")


class ValidatorFactory:
    """Factory class for creating validators dynamically with type safety.

    This class provides static methods for creating various types of validators
    and performing validation operations with comprehensive error reporting.

    The factory pattern enables consistent validator creation with proper
    configuration and composition, supporting both simple and complex
    validation scenarios.
    """

    @staticmethod
    def create_validator(
        validator_type: str, **kwargs: Dict[str, object]
    ) -> Callable[[object], bool]:
        """Creates a validator based on the specified type.

        Args:
            validator_type: The type of validator to create.
                Must be one of the supported validator types ('basic', 'composite').
            **kwargs: Additional parameters for the validator.
                Passed directly to the validator constructor.

        Returns:
            An instance of the requested validator.

        Raises:
            ValueError: If the validator type is unknown.

        Examples:
            >>> validator = ValidatorFactory.create_validator('basic')
            >>> validator(42)
            True
        """
        if validator_type == "basic":
            return BasicValidator(**kwargs)
        elif validator_type == "composite":
            return CompositeValidator(**kwargs)  # type: ignore
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")

    @staticmethod
    def validate_type(
        value: object,
        expected_type: Union[Type[T], Sequence[Type[object]]],
        path: str = "$",
        convert: bool = False,
    ) -> ValidationResult[T]:
        """Validate that a value matches the expected type recursively.

        Args:
            value: The value to check
            expected_type: Type or types to check against
            path: Current path in the validation (for error reporting)
            convert: Whether to attempt conversion to the expected type

        Returns:
            ValidationResult with validation status and details

        Examples:
            >>> result = ValidatorFactory.validate_type(42, int)
            >>> result.valid
            True

            >>> result = ValidatorFactory.validate_type("42", int, convert=True)
            >>> result.valid
            True
            >>> result.converted_value
            42
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
                return ValidationResult(True, [], None)  # type: ignore

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

                converted = ValidatorFactory._try_convert(value, typ)
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
        )

    @staticmethod
    def _try_convert(value: object, target_type: Type[V]) -> Optional[V]:
        """Try to convert a value to the target type.

        Args:
            value: Value to convert
            target_type: Type to convert to

        Returns:
            Converted value or None if conversion failed

        Note:
            This is an internal method used by validate_type.
        """
        try:
            # Handle built-in types
            if target_type is int:
                result = BasicValidator.safe_int_convert(value)
                return cast(V, result) if result is not None else None

            if target_type is bool:
                return cast(V, BasicValidator.safe_bool_convert(value))

            if target_type is float:
                result = BasicValidator.safe_float_convert(value)
                return cast(V, result) if result is not None else None

            if target_type is str:
                return cast(V, BasicValidator.safe_str_convert(value))

            # For other types, attempt direct construction
            if not isinstance(value, target_type):
                try:
                    # Use explicit constructor with proper type safety
                    return cast(V, target_type(value))  # type: ignore
                except (ValueError, TypeError):
                    pass

            return None
        except Exception:
            return None

    @staticmethod
    def validate_recursive(
        value: object,
        schema: SchemaTypeT,
        path: str = "$",
        convert: bool = False,
    ) -> ValidationResult[object]:
        """Recursively validate a value against a schema of arbitrary complexity.

        Args:
            value: Value to validate
            schema: Schema to validate against (dict, list/sequence, or type)
            path: Current path for error reporting
            convert: Whether to attempt type conversion

        Returns:
            ValidationResult with validation status and details

        Examples:
            >>> schema = {"name": str, "age": int}
            >>> data = {"name": "Alice", "age": 30}
            >>> result = ValidatorFactory.validate_recursive(data, schema)
            >>> result.valid
            True

            >>> data = {"name": "Alice", "age": "30"}
            >>> result = ValidatorFactory.validate_recursive(data, schema, convert=True)
            >>> result.valid
            True
            >>> result.converted_value  # doctest: +SKIP
            {'name': 'Alice', 'age': 30}
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
            return ValidatorFactory.validate_type(
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
                )
            # Cast the return value to ensure type compatibility
            schema_dict = cast(DictSchemaT, schema)
            value_dict = cast(Dict[str, object], value)
            return ValidatorFactory.validate_dict(
                value_dict, schema_dict, path, convert
            )

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
                )

            result: ValidationResult[List[object]] = ValidationResult(True, [], [])
            element_schema = schema[0]
            result_list: List[object] = []

            # Properly type the value as a sequence for iteration
            sequence_value = cast(Sequence[object], value)

            for i, item in enumerate(sequence_value):
                item_path = f"{path}[{i}]"
                item_result = ValidatorFactory.validate_recursive(
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
            type_result = ValidatorFactory.validate_type(value, schema, path, convert)
            return cast(ValidationResult[object], type_result)

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
            )

    @staticmethod
    def validate_dict(
        value: object,
        schema: DictSchemaT,
        path: str = "$",
        convert: bool = False,
        require_all_keys: bool = True,
    ) -> ValidationResult[Dict[str, object]]:
        """Validate that a dictionary conforms to a schema.

        Args:
            value: Dictionary to validate
            schema: Schema defining expected types for keys
            path: Current path for error reporting
            convert: Whether to attempt type conversion
            require_all_keys: Whether all schema keys must be present

        Returns:
            ValidationResult with validation status and converted dict if applicable

        Examples:
            >>> schema = {"name": str, "age": int}
            >>> data = {"name": "Alice", "age": 30}
            >>> result = ValidatorFactory.validate_dict(data, schema)
            >>> result.valid
            True

            >>> data = {"name": "Bob"}  # Missing 'age'
            >>> result = ValidatorFactory.validate_dict(data, schema)
            >>> result.valid
            False
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
                    expected_str = ValidatorFactory._get_type_name(expected_type)

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
                key_result = ValidatorFactory.validate_recursive(
                    dict_value[key], cast(SchemaTypeT, expected_type), key_path, convert
                )

                if key_result.valid and key_result.converted_value is not None:
                    result_dict[key] = key_result.converted_value
                else:
                    result_dict[key] = dict_value[key]

                result.merge(key_result)

        return result

    @staticmethod
    def _get_type_name(typ: object) -> str:
        """Get a human-readable name for a type or type collection.

        Args:
            typ: A type, tuple of types, or list/sequence of types

        Returns:
            A string representation of the type(s)
        """
        if isinstance(typ, type):
            return typ.__name__
        # Handle tuples of types (Union-like)
        elif isinstance(typ, tuple) and all(isinstance(t, type) for t in typ):
            return " | ".join(t.__name__ for t in typ)
        # Handle lists/sequences of types or values
        elif isinstance(typ, (list, tuple, Sequence)) and not isinstance(
            typ, (str, bytes)
        ):
            if len(typ) > 0:
                if isinstance(typ[0], type):
                    return f"List[{typ[0].__name__}]"
                else:
                    return "List[Dict]"  # For list of schema dicts
            else:
                return "List"  # Instead of List[Any]
        # Handle dictionaries (structural types)
        elif isinstance(typ, dict):
            return "Dict[str]"  # Instead of Dict[str, Any]
        else:
            return str(typ)
