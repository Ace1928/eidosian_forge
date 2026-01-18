type_forge.forge.type_forge
===========================

.. py:module:: type_forge.forge.type_forge

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   type_forge.forge.type_forge.TypeForge


Module Contents
---------------

.. py:class:: TypeForge

   Bases: :py:obj:`type_forge.core.base.TypeForgeBase`


   Core class for dynamic type creation, validation, and transformation.

   This class combines type validation, conversion, and dynamic type creation
   into a unified interface with recursive precision. It serves as the primary
   entry point for the type_forge module, providing a coherent API for all
   type-related operations.

   The system implements a recursive type validation system that can handle
   arbitrarily complex nested structures while maintaining full type safety
   and providing detailed error reporting.

   .. rubric:: Attributes

   types: Registry mapping type names to their class objects.
   validators: Collection of validators for the validation pipeline.

   .. rubric:: Examples

   >>> forge = TypeForge()
   >>> class Person: pass
   >>> forge.register_type("Person", Person)
   >>> person = forge.create_instance("Person", name="Alice", age=30)
   >>> forge.is_instance(person, "Person")
   True

   Initialize a new TypeForge instance with empty registries.

   Creates clean registries for types and validators that will be populated
   through the register_type and add_validator methods, establishing
   the foundation for dynamic type operations.


   .. py:method:: assert_type(value: object, expected_type: Type[type_forge.typing.definitions.R], message: Optional[type_forge.typing.definitions.ErrorMessage] = None) -> type_forge.typing.definitions.R
                  assert_type(value: object, expected_type: Tuple[Type[object], Ellipsis], message: Optional[type_forge.typing.definitions.ErrorMessage] = None) -> object

      Assert that a value has the expected type, raising TypeError if not.

      Performs strict type checking that raises an exception if the value
      doesn't match the expected type, enabling fail-fast behavior for
      critical type safety requirements.

      :param value: Value to check against the expected type.
                    Can be any object including None.
      :param expected_type: Type or tuple of types to check against.
                            Validation succeeds if value matches any of the specified types.
      :param message: Optional custom error message for the exception.
                      Defaults to None. If None, a detailed error message is generated.

      :returns: The original value with type guarantee if validation succeeds.

      :raises TypeError: If the value doesn't match the expected type.
          Contains either the custom message or detailed validation errors.

      .. rubric:: Examples

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

      .. seealso::

         :meth:`~TypeForge.check_type`: For boolean type checking without exceptions.
         :meth:`~TypeForge.validate_type`: For detailed validation results without exceptions.
         :class:`~..core.base.ValidationViolation`: For the structure of validation errors.



   .. py:method:: check_type(value, expected_type)

      Perform a simple type check without detailed reporting.

      Provides a simplified interface for type checking when only a boolean
      result is needed, without the detailed validation reporting of the
      validate_type method.

      :param value: Value to check against the expected type.
                    Can be any object including None.
      :param expected_type: Type or tuple of types to check against.
                            Validation succeeds if value matches any of the specified types.

      :returns: True if value matches the expected type, False otherwise.
      :rtype: bool

      .. rubric:: Examples

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

      .. seealso::

         :meth:`~TypeForge.validate_type`: For detailed validation results with violation information.
         :meth:`~TypeForge.assert_type`: For raising exceptions when type validation fails.



   .. py:method:: convert_value(value, target_type)

      Convert a value to the target type with detailed error tracking.

      Attempts to convert the given value to the specified target type,
      providing detailed information about success or failure.

      :param value: The value to convert.
      :param target_type: The type to convert the value to.

      :returns: A ConversionResult containing success status, converted value (if successful),
                and error message (if failed).

      .. rubric:: Examples

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



   .. py:method:: create_instance(name: type_forge.typing.definitions.TypeName, cls_type: Type[type_forge.typing.definitions.TInstance], *args: object, **kwargs: object) -> type_forge.typing.definitions.TInstance
                  create_instance(name: type_forge.typing.definitions.TypeName, *args: object, **kwargs: object) -> object

      Create an instance of a registered type with the provided arguments.

      Dynamically instantiates an object of the type associated with the given
      name, passing the provided arguments to its constructor. Provides a type-safe
      way to create objects from registered types.

      :param name: Name of the registered type to instantiate.
                   Must be previously registered with :meth:`~TypeForge.register_type`.
      :param cls_type: Optional first argument for type inference in the first overload.
                       When provided as the first argument, enables return type to be properly inferred.
      :param \*args: Positional arguments to pass to the constructor.
                     Will be passed directly to the type's __init__ method.
      :param \*\*kwargs: Keyword arguments to pass to the constructor.
                         Will be passed directly to the type's __init__ method.

      :returns: An instance of the registered type. If cls_type is provided as the first
                argument, the return type will be inferred as that type.

      :raises ValueError: If the requested type is not registered.
      :raises TypeError: If constructor arguments are incompatible with the type.

      .. rubric:: Examples

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



   .. py:method:: create_type(name, fields)

      Dynamically create a new type with the specified fields.

      Constructs a new type at runtime with the provided field definitions,
      registers it in the type registry, and returns the created type object.
      Enables programmatic type creation with structural validation.

      :param name: Name for the new type.
                   Must be unique within this forge instance.
      :param fields: Dictionary mapping field names to their types.
                     These will become attributes of the created type.

      :returns: The newly created type object, registered in the forge.
      :rtype: Type[object]

      :raises TypeCreationError: If type creation fails for any reason.
      :raises ValueError: If a type with the given name is already registered.

      .. rubric:: Examples

      >>> Person = forge.create_type("Person", {
      ...     "name": str,
      ...     "age": int
      ... })
      >>> person = Person()
      >>> person.name = "Alice"
      >>> person.age = 30



   .. py:method:: is_instance(value, type_name)

      Verify that an object is an instance of a registered type.

      Performs runtime type checking against a registered type, providing
      a simple boolean result indicating whether the instance matches the
      expected type.

      :param value: Object to validate.
                    Will be checked using isinstance() against the registered type.
      :param type_name: Name of the registered type to validate against.
                        Must be previously registered with :meth:`~TypeForge.register_type`.

      :returns: True if the instance matches the registered type, False otherwise.
      :rtype: bool

      :raises ValueError: If the requested type is not registered.

      .. rubric:: Examples

      >>> class Person: pass
      >>> forge.register_type("Person", Person)
      >>> person = Person()
      >>> forge.is_instance(person, "Person")
      True
      >>> forge.is_instance("not a person", "Person")
      False



   .. py:method:: register_type(name, cls)

      Register a type in the forge's type registry.

      Associates a name with a type class in the registry, making it available
      for dynamic instantiation and validation operations. Prevents duplicate
      registrations to maintain registry integrity.

      :param name: Identifier for the type in the registry.
                   Must be unique within this forge instance.
      :param cls: Class object to associate with the name.
                  Will be stored for later instantiation and validation.

      :raises ValueError: If a type with the given name is already registered.

      .. rubric:: Examples

      >>> forge = TypeForge()
      >>> class Person: pass
      >>> forge.register_type("Person", Person)

      .. note::

         Registered types become accessible through all forge operations
         including :meth:`~TypeForge.create_instance` and :meth:`~TypeForge.is_instance`.



   .. py:method:: safe_convert(value, target_type, default = None)

      Safely convert a value to the target type, returning default on failure.

      Attempts to convert the value to the specified type, but returns
      a default value if conversion fails instead of raising an exception.

      :param value: The value to convert.
      :param target_type: The type to convert the value to.
      :param default: The default value to return if conversion fails.
                      Defaults to None.

      :returns: The converted value or the default value if conversion fails.

      .. rubric:: Examples

      >>> forge = TypeForge()
      >>> forge.safe_convert("42", int)
      42
      >>> forge.safe_convert("not_a_number", int, 0)
      0



   .. py:method:: validate(value)

      Validate a value using registered validators.

      Implements the base class validate method to check if a value
      passes validation through all registered validators. This method
      follows the Liskov Substitution Principle by maintaining the exact
      same signature as the base class method.

      :param value: Object to validate using the registered validators.

      :returns: True if the value passes all validators, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> forge = TypeForge()
      >>> # Add validators to the forge...
      >>> forge.validate(42)
      True

      .. seealso:: :meth:`~TypeForge.is_instance`: For checking if a value is an instance of a registered type.



   .. py:method:: validate_and_convert(value, target_type, path = '$')

      Validate and convert a value to the target type in a single operation.

      Combines validation and conversion into a unified operation, ensuring
      type safety while attempting to transform the value into the expected type.
      Provides detailed results including conversion status and violations.

      :param value: Value to validate and convert.
                    Can be any object that might be convertible to target_type.
      :param target_type: Type to convert the value to.
                          Must be a valid Python type that supports conversion from value.
      :param path: JSON path-like string for contextual error reporting.
                   Defaults to "$", representing the root of the validation tree.

      :returns:

                Result object containing:
                    - valid (bool): Whether validation and conversion succeeded
                    - value (R, optional): The converted value with type guarantee
                    - violations (List[ValidationViolation], optional): Details on conversion failures
      :rtype: ValidationResult[R]

      .. rubric:: Examples

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

      .. seealso::

         :meth:`~TypeForge.validate_type`: The underlying method used for validation with conversion.
         :class:`~..core.base.ValidationResult`: For the structure of the returned result.



   .. py:method:: validate_dict_schema(data, schema, convert = False, require_all_keys = True)
      :staticmethod:


      Validate that a dictionary conforms to a schema with field type checking.

      Performs structural validation of a dictionary against a schema definition,
      ensuring field types match expectations and optionally converting values
      or requiring all schema keys to be present.

      :param data: Dictionary-like object to validate.
                   Expected to be a dict or dict-like with key access; validated at runtime.
      :param schema: Schema defining expected types for dictionary keys.
                     A dictionary mapping key names to their expected types.
      :param convert: Whether to attempt type conversion for mismatched types.
                      Defaults to False. When True, will attempt to convert values to match schema types.
      :param require_all_keys: Whether all schema keys must be present in the data.
                               Defaults to True. When False, missing keys will not cause validation failure.

      :returns:

                Result object containing:
                    - valid (bool): Whether validation succeeded
                    - value (Dict[str, object], optional): The validated (and possibly converted) dictionary
                    - violations (List[ValidationViolation], optional): Details on validation failures
      :rtype: ValidationResult[Dict[str, object]]

      .. rubric:: Examples

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

      .. seealso::

         :meth:`~TypeForge.validate_type`: For validating individual values.
         :meth:`~TypeForge.validate_recursive`: For validating deeply nested structures.



   .. py:method:: validate_recursive(value, schema, path = '$', convert = False)
      :staticmethod:


      Recursively validate a value against a schema of arbitrary complexity.

      Performs deep structural validation of nested data structures against
      complex schema definitions, supporting arbitrary nesting depth with
      precise path tracking for error reporting.

      :param value: Value to validate against the schema.
                    Can be any object, including nested structures like dicts and lists.
      :param schema: Schema definition of arbitrary complexity.
                     Can include nested dictionaries, lists of types, and primitive types.
                     Must be one of: Type, Tuple[Type, ...], Dict[str, SchemaValueT], List[SchemaTypeT]
      :param path: JSON path-like string for contextual error reporting.
                   Defaults to "$", representing the root of the validation tree.
      :param convert: Whether to attempt type conversion for mismatched types.
                      Defaults to False. When True, will attempt to convert values to match schema types.

      :returns:

                Result object containing:
                    - valid (bool): Whether validation succeeded
                    - converted_value (object, optional): The validated (and possibly converted) structure
                    - violations (List[TypeViolation], optional): Details on validation failures
      :rtype: ValidationResult[object]

      .. rubric:: Examples

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

      .. seealso::

         :meth:`~TypeForge.validate_dict_schema`: For validating dictionary-specific schemas.
         :class:`SchemaTypeT`: For the schema type definition.



   .. py:method:: validate_type(value, expected_type, path = '$', convert = False)
      :staticmethod:


      Validate that a value matches the expected type with detailed reporting.

      Performs deep validation of a value against expected types, optionally
      attempting type conversion. Provides detailed validation results including
      success status and any validation violations.

      :param value: Value to validate against the expected type.
                    Can be any object including None.
      :param expected_type: Single type or sequence of types to check against.
                            For sequences, the validation succeeds if the value matches any of the types.
      :param path: JSON path-like string for contextual error reporting.
                   Defaults to "$", representing the root of the validation tree.
      :param convert: Whether to attempt type conversion if validation fails.
                      Defaults to False. When True, will attempt to convert value to expected_type.

      :returns:

                Result object containing:
                    - valid (bool): Whether validation succeeded
                    - value (R, optional): The validated (and possibly converted) value
                    - violations (List[ValidationViolation], optional): Details on validation failures
      :rtype: ValidationResult[R]

      .. rubric:: Examples

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



   .. py:attribute:: types
      :type:  type_forge.typing.definitions.TypeRegistry


