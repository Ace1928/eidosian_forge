"""
Type Forge Core Implementation Module

This module contains the foundational functionality for the type forging process,
implementing dynamic type creation, validation, and transformation with recursive
precision and structural integrity.

The TypeForge class provides a unified interface for complex type operations:
- Dynamic type registration and instantiation
- Strict type validation with detailed reporting
- Recursive schema validation for nested structures
- Type conversion with safety guarantees

All operations maintain full type safety through compile-time checks and
runtime validation, ensuring system-wide integrity.
"""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from type_forge.core import TypeCreationError, TypeForgeBase, ValidationResult
from type_forge.typing import (
    ConversionResult,
    DictSchemaT,
    ErrorMessage,
    FieldDefinitions,
    R,
    SchemaTypeT,
    T,
    TInstance,
    TypeName,
    TypeRegistry,
    U,
    ValidationPath,
)
from type_forge.validators import ValidatorFactory

# Author and version information
__author__ = "TypeForge Team"
__version__ = "0.1.0"


class TypeForge(TypeForgeBase):
    """Core class for dynamic type creation, validation, and transformation.

    This class combines type validation, conversion, and dynamic type creation
    into a unified interface with recursive precision. It serves as the primary
    entry point for the type_forge module, providing a coherent API for all
    type-related operations.

    The system implements a recursive type validation system that can handle
    arbitrarily complex nested structures while maintaining full type safety
    and providing detailed error reporting.

    Attributes:
        types: Registry mapping type names to their class objects.
        validators: Collection of validators for the validation pipeline.

    Examples:
        >>> forge = TypeForge()
        >>> class Person: pass
        >>> forge.register_type("Person", Person)
        >>> person = forge.create_instance("Person", name="Alice", age=30)
        >>> forge.is_instance(person, "Person")
        True
    """

    def __init__(self) -> None:
        """Initialize a new TypeForge instance with empty registries.

        Creates clean registries for types and validators that will be populated
        through the register_type and add_validator methods, establishing
        the foundation for dynamic type operations.
        """
        super().__init__()
        self.types: TypeRegistry = {}

    def register_type(self, name: TypeName, cls: Type[T]) -> None:
        """Register a type in the forge's type registry.

        Associates a name with a type class in the registry, making it available
        for dynamic instantiation and validation operations. Prevents duplicate
        registrations to maintain registry integrity.

        Args:
            name: Identifier for the type in the registry.
                Must be unique within this forge instance.
            cls: Class object to associate with the name.
                Will be stored for later instantiation and validation.

        Raises:
            ValueError: If a type with the given name is already registered.

        Examples:
            >>> forge = TypeForge()
            >>> class Person: pass
            >>> forge.register_type("Person", Person)

        Note:
            Registered types become accessible through all forge operations
            including :meth:`~TypeForge.create_instance` and :meth:`~TypeForge.is_instance`.
        """
        if name in self.types:
            raise ValueError(f"Type '{name}' is already registered.")
        self.types[name] = cls

    # Fix for overlapping overloads - use a generic for the first one
    @overload
    def create_instance(
        self,
        name: TypeName,
        cls_type: Type[TInstance],
        *args: object,
        **kwargs: object,
    ) -> TInstance: ...

    @overload
    def create_instance(
        self,
        name: TypeName,
        *args: object,
        **kwargs: object,
    ) -> object: ...

    def create_instance(
        self,
        name: TypeName,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Create an instance of a registered type with the provided arguments.

        Dynamically instantiates an object of the type associated with the given
        name, passing the provided arguments to its constructor. Provides a type-safe
        way to create objects from registered types.

        Args:
            name: Name of the registered type to instantiate.
                Must be previously registered with :meth:`~TypeForge.register_type`.
            cls_type: Optional first argument for type inference in the first overload.
                When provided as the first argument, enables return type to be properly inferred.
            *args: Positional arguments to pass to the constructor.
                Will be passed directly to the type's __init__ method.
            **kwargs: Keyword arguments to pass to the constructor.
                Will be passed directly to the type's __init__ method.

        Returns:
            An instance of the registered type. If cls_type is provided as the first
            argument, the return type will be inferred as that type.

        Raises:
            ValueError: If the requested type is not registered.
            TypeError: If constructor arguments are incompatible with the type.

        Examples:
            >>> class Person:
            ...     def __init__(self, name: str, age: int):
            ...         self.name = name
            ...         self.age = age
            >>> forge.register_type("Person", Person)
            >>> person = forge.create_instance("Person", name="Alice", age=30)
            >>> person.name
            'Alice'

            >>> # With type inference
            >>> from typing import TypeVar
            >>> T = TypeVar('T', bound=Person)
            >>> person_typed = forge.create_instance("Person", Person, name="Bob", age=25)
            >>> # person_typed will have proper type inference as Person
        """
        if name not in self.types:
            raise ValueError(f"Type '{name}' is not registered.")

        # Extract cls_type if it's the first arg and a type
        if args and isinstance(args[0], type):
            # We have a cls_type as first arg, use remaining args
            cls = self.types[name]
            return cls(*args[1:], **kwargs)
        # No cls_type, use all args
        cls = self.types[name]
        return cls(*args, **kwargs)

    def validate(self, value: object) -> bool:
        """Validate a value using registered validators.

        Implements the base class validate method to check if a value
        passes validation through all registered validators. This method
        follows the Liskov Substitution Principle by maintaining the exact
        same signature as the base class method.

        Args:
            value: Object to validate using the registered validators.

        Returns:
            bool: True if the value passes all validators, False otherwise.

        Examples:
            >>> forge = TypeForge()
            >>> # Add validators to the forge...
            >>> forge.validate(42)
            True

        See Also:
            :meth:`~TypeForge.is_instance`: For checking if a value is an instance of a registered type.
        """
        # Standard implementation calling the base class method
        return super().validate(value)

    def is_instance(self, value: object, type_name: TypeName) -> bool:
        """Verify that an object is an instance of a registered type.

        Performs runtime type checking against a registered type, providing
        a simple boolean result indicating whether the instance matches the
        expected type.

        Args:
            value: Object to validate.
                Will be checked using isinstance() against the registered type.
            type_name: Name of the registered type to validate against.
                Must be previously registered with :meth:`~TypeForge.register_type`.

        Returns:
            bool: True if the instance matches the registered type, False otherwise.

        Raises:
            ValueError: If the requested type is not registered.

        Examples:
            >>> class Person: pass
            >>> forge.register_type("Person", Person)
            >>> person = Person()
            >>> forge.is_instance(person, "Person")
            True
            >>> forge.is_instance("not a person", "Person")
            False
        """
        if type_name not in self.types:
            raise ValueError(f"Type '{type_name}' is not registered.")
        return isinstance(value, self.types[type_name])

    def create_type(self, name: TypeName, fields: FieldDefinitions) -> Type[object]:
        """Dynamically create a new type with the specified fields.

        Constructs a new type at runtime with the provided field definitions,
        registers it in the type registry, and returns the created type object.
        Enables programmatic type creation with structural validation.

        Args:
            name: Name for the new type.
                Must be unique within this forge instance.
            fields: Dictionary mapping field names to their types.
                These will become attributes of the created type.

        Returns:
            Type[object]: The newly created type object, registered in the forge.

        Raises:
            TypeCreationError: If type creation fails for any reason.
            ValueError: If a type with the given name is already registered.

        Examples:
            >>> Person = forge.create_type("Person", {
            ...     "name": str,
            ...     "age": int
            ... })
            >>> person = Person()
            >>> person.name = "Alice"
            >>> person.age = 30
        """
        if name in self.types:
            raise ValueError(f"Type '{name}' is already registered.")

        try:
            # Create a new type with the specified fields as class attributes
            new_type = type(name, (), fields)
            self.register_type(name, new_type)
            return new_type
        except Exception as e:
            raise TypeCreationError(f"Failed to create type '{name}': {str(e)}") from e

    @staticmethod
    def validate_type(
        value: object,
        expected_type: Union[Type[R], Sequence[Type[object]]],
        path: ValidationPath = "$",
        convert: bool = False,
    ) -> ValidationResult[R]:
        """Validate that a value matches the expected type with detailed reporting.

        Performs deep validation of a value against expected types, optionally
        attempting type conversion. Provides detailed validation results including
        success status and any validation violations.

        Args:
            value: Value to validate against the expected type.
                Can be any object including None.
            expected_type: Single type or sequence of types to check against.
                For sequences, the validation succeeds if the value matches any of the types.
            path: JSON path-like string for contextual error reporting.
                Defaults to "$", representing the root of the validation tree.
            convert: Whether to attempt type conversion if validation fails.
                Defaults to False. When True, will attempt to convert value to expected_type.

        Returns:
            ValidationResult[R]: Result object containing:
                - valid (bool): Whether validation succeeded
                - value (R, optional): The validated (and possibly converted) value
                - violations (List[ValidationViolation], optional): Details on validation failures

        Examples:
            >>> # Basic validation
            >>> result = TypeForge.validate_type(42, int)
            >>> result.valid
            True

            >>> # Type conversion
            >>> result = TypeForge.validate_type("42", int, convert=True)
            >>> result.valid
            True
            >>> result.value
            42

            >>> # Multiple allowed types
            >>> result = TypeForge.validate_type(42, (str, int))
            >>> result.valid
            True
        """
        # Using a more explicit approach to handle the generic type R properly
        if isinstance(expected_type, type):
            # Single type case
            return cast(
                ValidationResult[R],
                ValidatorFactory.validate_type(value, expected_type, path, convert),
            )
        # Sequence of types case with proper typing
        return cast(
            ValidationResult[R],
            ValidatorFactory.validate_type(value, expected_type, path, convert),
        )

    @staticmethod
    def validate_dict_schema(
        data: object,
        schema: Mapping[str, SchemaTypeT[Any]],
        convert: bool = False,
        require_all_keys: bool = True,
    ) -> ValidationResult[Dict[str, object]]:
        """Validate that a dictionary conforms to a schema with field type checking.

        Performs structural validation of a dictionary against a schema definition,
        ensuring field types match expectations and optionally converting values
        or requiring all schema keys to be present.

        Args:
            data: Dictionary-like object to validate.
                Expected to be a dict or dict-like with key access; validated at runtime.
            schema: Schema defining expected types for dictionary keys.
                A dictionary mapping key names to their expected types.
            convert: Whether to attempt type conversion for mismatched types.
                Defaults to False. When True, will attempt to convert values to match schema types.
            require_all_keys: Whether all schema keys must be present in the data.
                Defaults to True. When False, missing keys will not cause validation failure.

        Returns:
            ValidationResult[Dict[str, object]]: Result object containing:
                - valid (bool): Whether validation succeeded
                - value (Dict[str, object], optional): The validated (and possibly converted) dictionary
                - violations (List[ValidationViolation], optional): Details on validation failures

        Examples:
            >>> person_schema = {"name": str, "age": int}
            >>> # Valid data
            >>> result = TypeForge.validate_dict_schema(
            ...     {"name": "Alice", "age": 30},
            ...     person_schema
            ... )
            >>> result.valid
            True

            >>> # Type conversion
            >>> result = TypeForge.validate_dict_schema(
            ...     {"name": "Alice", "age": "30"},
            ...     person_schema,
            ...     convert=True
            ... )
            >>> result.valid
            True
            >>> result.value
            {'name': 'Alice', 'age': 30}

            >>> # Missing key with require_all_keys=True
            >>> result = TypeForge.validate_dict_schema(
            ...     {"name": "Alice"},
            ...     person_schema,
            ...     require_all_keys=True
            ... )
            >>> result.valid
            False

        See Also:
            :meth:`~TypeForge.validate_type`: For validating individual values.
            :meth:`~TypeForge.validate_recursive`: For validating deeply nested structures.
        """
        # Explicit cast to ensure type safety with the factory method
        # We're using a properly typed schema that's compatible with DictSchemaT
        return ValidatorFactory.validate_dict(
            data,
            cast(DictSchemaT, schema),
            path="$",
            convert=convert,
            require_all_keys=require_all_keys,
        )

    @staticmethod
    def validate_recursive(
        value: object,
        schema: Union[
            Type[object],
            Tuple[Type[object], ...],
            Dict[str, object],
            List[object],
        ],
        path: str = "$",
        convert: bool = False,
    ) -> ValidationResult[object]:
        """Recursively validate a value against a schema of arbitrary complexity.

        Performs deep structural validation of nested data structures against
        complex schema definitions, supporting arbitrary nesting depth with
        precise path tracking for error reporting.

        Args:
            value: Value to validate against the schema.
                Can be any object, including nested structures like dicts and lists.
            schema: Schema definition of arbitrary complexity.
                Can include nested dictionaries, lists of types, and primitive types.
                Must be one of: Type, Tuple[Type, ...], Dict[str, SchemaValueT], List[SchemaTypeT]
            path: JSON path-like string for contextual error reporting.
                Defaults to "$", representing the root of the validation tree.
            convert: Whether to attempt type conversion for mismatched types.
                Defaults to False. When True, will attempt to convert values to match schema types.

        Returns:
            ValidationResult[object]: Result object containing:
                - valid (bool): Whether validation succeeded
                - converted_value (object, optional): The validated (and possibly converted) structure
                - violations (List[TypeViolation], optional): Details on validation failures

        Examples:
            >>> # Nested schema validation
            >>> nested_schema = {
            ...     "user": {
            ...         "profile": {"name": str, "age": int},
            ...         "settings": {"theme": str, "notifications": bool}
            ...     }
            ... }
            >>> complex_data = {
            ...     "user": {
            ...         "profile": {"name": "Alice", "age": 30},
            ...         "settings": {"theme": "dark", "notifications": True}
            ...     }
            ... }
            >>> result = TypeForge.validate_recursive(complex_data, nested_schema)
            >>> result.valid
            True

            >>> # Invalid nested data
            >>> invalid_data = {
            ...     "user": {
            ...         "profile": {"name": "Bob", "age": "thirty"},  # Age should be int
            ...         "settings": {"theme": "light"}  # Missing notifications
            ...     }
            ... }
            >>> result = TypeForge.validate_recursive(invalid_data, nested_schema)
            >>> result.valid
            False

        See Also:
            :meth:`~TypeForge.validate_dict_schema`: For validating dictionary-specific schemas.
            :class:`SchemaTypeT`: For the schema type definition.
        """
        # Precise typing ensures we're passing the correct schema type to the validator
        # Using cast to ensure type safety with the factory method
        return ValidatorFactory.validate_recursive(
            value,
            cast(SchemaTypeT[object], schema),
            path,
            convert,
        )

    def validate_and_convert(
        self,
        value: object,
        target_type: Type[R],
        path: ValidationPath = "$",
    ) -> ValidationResult[R]:
        """Validate and convert a value to the target type in a single operation.

        Combines validation and conversion into a unified operation, ensuring
        type safety while attempting to transform the value into the expected type.
        Provides detailed results including conversion status and violations.

        Args:
            value: Value to validate and convert.
                Can be any object that might be convertible to target_type.
            target_type: Type to convert the value to.
                Must be a valid Python type that supports conversion from value.
            path: JSON path-like string for contextual error reporting.
                Defaults to "$", representing the root of the validation tree.

        Returns:
            ValidationResult[R]: Result object containing:
                - valid (bool): Whether validation and conversion succeeded
                - value (R, optional): The converted value with type guarantee
                - violations (List[ValidationViolation], optional): Details on conversion failures

        Examples:
            >>> # String to integer conversion
            >>> result = forge.validate_and_convert("42", int)
            >>> result.valid
            True
            >>> result.value
            42

            >>> # Failed conversion
            >>> result = forge.validate_and_convert("not_an_int", int)
            >>> result.valid
            False
            >>> str(result.violations[0])  # doctest: +SKIP
            "At path '$': Expected int, found 'str' (wrong_type)"

        See Also:
            :meth:`~TypeForge.validate_type`: The underlying method used for validation with conversion.
            :class:`~..core.base.ValidationResult`: For the structure of the returned result.
        """
        return self.validate_type(value, target_type, path, convert=True)

    def check_type(
        self,
        value: object,
        expected_type: Union[Type[U], Tuple[Type[object], ...]],
    ) -> bool:
        """Perform a simple type check without detailed reporting.

        Provides a simplified interface for type checking when only a boolean
        result is needed, without the detailed validation reporting of the
        validate_type method.

        Args:
            value: Value to check against the expected type.
                Can be any object including None.
            expected_type: Type or tuple of types to check against.
                Validation succeeds if value matches any of the specified types.

        Returns:
            bool: True if value matches the expected type, False otherwise.

        Examples:
            >>> # Simple type check
            >>> forge = TypeForge()
            >>> forge.check_type("hello", str)
            True

            >>> # Multiple allowed types
            >>> forge.check_type(42, (str, int))
            True

            >>> # Failed type check
            >>> forge.check_type(42, str)
            False

        See Also:
            :meth:`~TypeForge.validate_type`: For detailed validation results with violation information.
            :meth:`~TypeForge.assert_type`: For raising exceptions when type validation fails.
        """
        return self.validate_type(value, expected_type).valid

    @overload
    def assert_type(
        self,
        value: object,
        expected_type: Type[R],
        message: Optional[ErrorMessage] = None,
    ) -> R: ...

    @overload
    def assert_type(
        self,
        value: object,
        expected_type: Tuple[Type[object], ...],
        message: Optional[ErrorMessage] = None,
    ) -> object: ...

    def assert_type(
        self,
        value: object,
        expected_type: Union[Type[R], Tuple[Type[object], ...]],
        message: Optional[ErrorMessage] = None,
    ) -> object:
        """Assert that a value has the expected type, raising TypeError if not.

        Performs strict type checking that raises an exception if the value
        doesn't match the expected type, enabling fail-fast behavior for
        critical type safety requirements.

        Args:
            value: Value to check against the expected type.
                Can be any object including None.
            expected_type: Type or tuple of types to check against.
                Validation succeeds if value matches any of the specified types.
            message: Optional custom error message for the exception.
                Defaults to None. If None, a detailed error message is generated.

        Returns:
            The original value with type guarantee if validation succeeds.

        Raises:
            TypeError: If the value doesn't match the expected type.
                Contains either the custom message or detailed validation errors.

        Examples:
            >>> # Successful type assertion
            >>> forge = TypeForge()
            >>> age = forge.assert_type(42, int)
            >>> age

            >>> # Failed type assertion with default error
            >>> try:
            ...     forge.assert_type("hello", int)
            ... except TypeError as e:
            ...     "Type assertion failed" in str(e)
            True

        See Also:
            :meth:`~TypeForge.check_type`: For boolean type checking without exceptions.
            :meth:`~TypeForge.validate_type`: For detailed validation results without exceptions.
            :class:`~..core.base.ValidationViolation`: For the structure of validation errors.
        """
        result = self.validate_type(value, expected_type)
        if not result.valid:
            error_msg = message if message is not None else ""
            if not error_msg:
                violations = "\n".join(str(v) for v in result.violations)
                error_msg = f"Type assertion failed: {violations}"
            raise TypeError(error_msg)

        # We can safely cast here as we've verified the type
        return value

    # Type conversion utilities for common types

    def convert_value(self, value: object, target_type: Type[R]) -> ConversionResult[R]:
        """Convert a value to the target type with detailed error tracking.

        Attempts to convert the given value to the specified target type,
        providing detailed information about success or failure.

        Args:
            value: The value to convert.
            target_type: The type to convert the value to.

        Returns:
            A ConversionResult containing success status, converted value (if successful),
            and error message (if failed).

        Examples:
            >>> forge = TypeForge()
            >>> result = forge.convert_value("42", int)
            >>> result.success
            True
            >>> result.value
            42

            >>> result = forge.convert_value("not_a_number", int)
            >>> result.success
            False
            >>> result.error is not None
            True
        """
        # Import here to avoid circular imports
        from type_forge.typing.conversion import try_convert

        return try_convert(value, target_type)

    def safe_convert(
        self,
        value: object,
        target_type: Type[R],
        default: Optional[R] = None,
    ) -> Optional[R]:
        """Safely convert a value to the target type, returning default on failure.

        Attempts to convert the value to the specified type, but returns
        a default value if conversion fails instead of raising an exception.

        Args:
            value: The value to convert.
            target_type: The type to convert the value to.
            default: The default value to return if conversion fails.
                Defaults to None.

        Returns:
            The converted value or the default value if conversion fails.

        Examples:
            >>> forge = TypeForge()
            >>> forge.safe_convert("42", int)
            42
            >>> forge.safe_convert("not_a_number", int, 0)
            0
        """
        result = self.convert_value(value, target_type)
        if result.success and result.value is not None:
            return result.value
        return default
