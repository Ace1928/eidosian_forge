.. py:module:: src.type_forge.validators

This module initializes the validators submodule of the type_forge package.

   It provides a centralized interface for importing various validators, allowing for
   modular design and easy access to validation functionalities.

   Available Validators:
   - Basic Validators
   - Composite Validators
   - Validator Factory


Package Contents
----------------


   .. py:class:: BaseValidator   :module: 

      Base class for all validators in the type_forge framework.

      Provides the fundamental validation interface that all validators must implement,
      with support for both simple boolean validation and detailed validation results.

      Validators form the core of the type forge validation process, each implementing
      specific validation logic while adhering to a common interface that enables
      composition and chaining. This follows the "Composition Over Inheritance"
      principle from Eidosian design.

      The class follows the Template Method pattern, providing a default implementation
      of validate_with_detail that builds upon the abstract validate method
      that subclasses must implement.




   .. py:class:: BaseValidator   :module: 

      Base class for all validators in the type_forge framework.

      Provides the fundamental validation interface that all validators must implement,
      with support for both simple boolean validation and detailed validation results.

      Validators form the core of the type forge validation process, each implementing
      specific validation logic while adhering to a common interface that enables
      composition and chaining. This follows the "Composition Over Inheritance"
      principle from Eidosian design.

      The class follows the Template Method pattern, providing a default implementation
      of validate_with_detail that builds upon the abstract validate method
      that subclasses must implement.




   .. py:class:: BasicValidator   :module: 

      A class for basic data type validators.




   .. py:class:: BasicValidator   :module: 

      A class for basic data type validators.




   .. py:class:: CompositeValidator(validators)   :module: 

      A validator that combines multiple validators.

      Initialize a composite validator with a list of validator functions.

      :param validators: List of validator functions that take a value and return a boolean




   .. py:class:: CompositeValidator(validators)   :module: 

      A validator that combines multiple validators.

      Initialize a composite validator with a list of validator functions.

      :param validators: List of validator functions that take a value and return a boolean




   .. py:class:: SupportsBoolConversion   :module: 

      Protocol for types that can be converted to bool.

      This protocol defines the interface for objects that support
      conversion to boolean values through the __bool__ method.

      .. rubric:: Examples

      >>> class CustomBoolean:
      ...     def __init__(self, value: bool) -> None:
      ...         self.value = value
      ...     def __bool__(self) -> bool:
      ...         return self.value
      >>> bool(CustomBoolean(True))  # True




   .. py:class:: SupportsFloatConversion   :module: 

      Protocol for types that can be converted to float.

      This protocol defines the interface for objects that support
      conversion to floating-point values through the __float__ method.

      .. rubric:: Examples

      >>> class CustomFloat:
      ...     def __init__(self, value: float) -> None:
      ...         self.value = value
      ...     def __float__(self) -> float:
      ...         return self.value
      >>> float(CustomFloat(3.14))  # 3.14




   .. py:class:: SupportsIntConversion   :module: 

      Protocol for types that can be converted to int.

      This protocol defines the interface for objects that support
      conversion to integer values through the __int__ method.

      .. rubric:: Examples

      >>> class CustomInteger:
      ...     def __init__(self, value: int) -> None:
      ...         self.value = value
      ...     def __int__(self) -> int:
      ...         return self.value
      >>> isinstance(CustomInteger(42), SupportsIntConversion)  # True at runtime




   .. py:class:: TypeViolation   :module: 

      Immutable record of a type violation with path tracking.

      Provides a structured representation of a type violation with
      context information for precise error reporting and diagnosis.
      The frozen dataclass ensures immutability for safer error handling.

      .. rubric:: Attributes

      path: JSON path to the location of the violation.
      expected: Description of expected type or value.
      found: Description of actual type or value found.
      kind: Category of violation from TypeViolationKind.

      .. rubric:: Examples

      ```python
      violation = TypeViolation(
          path="user.address.zipcode",
          expected="string of 5 digits",
          found="'ABC123'",
          kind=TypeViolationKind.INVALID_VALUE
      )
      ```




   .. py:class:: TypeViolation   :module: 

      Immutable record of a type violation with path tracking.

      Provides a structured representation of a type violation with
      context information for precise error reporting and diagnosis.
      The frozen dataclass ensures immutability for safer error handling.

      .. rubric:: Attributes

      path: JSON path to the location of the violation.
      expected: Description of expected type or value.
      found: Description of actual type or value found.
      kind: Category of violation from TypeViolationKind.

      .. rubric:: Examples

      ```python
      violation = TypeViolation(
          path="user.address.zipcode",
          expected="string of 5 digits",
          found="'ABC123'",
          kind=TypeViolationKind.INVALID_VALUE
      )
      ```




   .. py:class:: TypeViolation   :module: 

      Immutable record of a type violation with path tracking.

      Provides a structured representation of a type violation with
      context information for precise error reporting and diagnosis.
      The frozen dataclass ensures immutability for safer error handling.

      .. rubric:: Attributes

      path: JSON path to the location of the violation.
      expected: Description of expected type or value.
      found: Description of actual type or value found.
      kind: Category of violation from TypeViolationKind.

      .. rubric:: Examples

      ```python
      violation = TypeViolation(
          path="user.address.zipcode",
          expected="string of 5 digits",
          found="'ABC123'",
          kind=TypeViolationKind.INVALID_VALUE
      )
      ```




   .. py:class:: TypeViolationKind(*args, **kwds)   :module: 

      Enumeration of possible type violation categories.

      Provides a structured taxonomy of type violations for precise
      error categorization and handling.

      .. rubric:: Attributes

      WRONG_TYPE: Value has incorrect type.
      MISSING_KEY: Required key is absent.
      INVALID_VALUE: Value fails validation constraints.
      SCHEMA_MISMATCH: Value structure doesn't match schema.
      CONVERSION_ERROR: Type conversion failed.

      .. rubric:: Examples

      ```python
      if isinstance(value, str):
          return TypeViolationKind.WRONG_TYPE
      ```




   .. py:class:: TypeViolationKind(*args, **kwds)   :module: 

      Enumeration of possible type violation categories.

      Provides a structured taxonomy of type violations for precise
      error categorization and handling.

      .. rubric:: Attributes

      WRONG_TYPE: Value has incorrect type.
      MISSING_KEY: Required key is absent.
      INVALID_VALUE: Value fails validation constraints.
      SCHEMA_MISMATCH: Value structure doesn't match schema.
      CONVERSION_ERROR: Type conversion failed.

      .. rubric:: Examples

      ```python
      if isinstance(value, str):
          return TypeViolationKind.WRONG_TYPE
      ```




   .. py:class:: TypeViolationKind(*args, **kwds)   :module: 

      Enumeration of possible type violation categories.

      Provides a structured taxonomy of type violations for precise
      error categorization and handling.

      .. rubric:: Attributes

      WRONG_TYPE: Value has incorrect type.
      MISSING_KEY: Required key is absent.
      INVALID_VALUE: Value fails validation constraints.
      SCHEMA_MISMATCH: Value structure doesn't match schema.
      CONVERSION_ERROR: Type conversion failed.

      .. rubric:: Examples

      ```python
      if isinstance(value, str):
          return TypeViolationKind.WRONG_TYPE
      ```




   .. py:class:: ValidationIssue(severity, message, path = None, context = None)   :module: 

      Detailed representation of a validation issue with context and severity.

      This class encapsulates information about a validation issue, including
      its severity, location, message, and contextual information for debugging
      and correction.

      .. rubric:: Attributes

      severity (ValidationSeverity): The severity level of the issue
      message (str): Human-readable description of the issue
      path (Optional[str]): Path to the location of the issue (e.g., "user.address.city")
      context (Dict[str, object]): Additional contextual information about the issue

      .. rubric:: Examples

      >>> issue = ValidationIssue(ValidationSeverity.ERROR, "Invalid email format")
      >>> issue.severity.name
      'ERROR'
      >>> issue.message
      'Invalid email format'
      >>> detailed = ValidationIssue(
      ...     ValidationSeverity.WARNING,
      ...     "Value outside recommended range",
      ...     path="settings.timeout",
      ...     context={"value": 120, "recommended_max": 60}
      ... )
      >>> detailed.path
      'settings.timeout'

      Initialize a ValidationIssue.

      :param severity: The severity level of the issue
      :param message: Human-readable description of the issue
      :param path: Path to the location of the issue (e.g., "user.address.city")
      :param context: Additional contextual information about the issue




   .. py:class:: ValidationReport   :module: 

      Comprehensive report of validation results including all issues found.

      This class collects and organizes validation issues, providing methods
      to query and analyze validation results in detail.

      .. rubric:: Attributes

      issues (List[ValidationIssue]): List of all validation issues found

      .. rubric:: Examples

      >>> report = ValidationReport()
      >>> report.add_error("Invalid email")
      >>> report.add_warning("Name unusually short", path="user.name")
      >>> report.is_valid()
      False
      >>> report.has_warnings()
      True
      >>> len(report.get_issues())
      2

      Initialize an empty ValidationReport.




   .. py:class:: ValidationResult   :module: 

      Result of type validation with possible conversion.

      This class encapsulates the outcome of a validation operation, including
      whether validation passed, any violations that occurred, and an optional
      converted value that maintains its type through generic constraints.

      The ValidationResult maintains type safety through covariant generics,
      ensuring that type information flows correctly through validation chains
      and transformations. It acts as both a container for validation status
      and a monad-like structure that can be composed and transformed while
      preserving the validation context.

      .. rubric:: Attributes

      valid (bool): Boolean indicating if validation succeeded
      violations (List[TypeViolation]): List of specific type violations encountered
      converted_value (Optional[T]): Optional transformed value that maintains
          its type through generics

      .. rubric:: Examples

      >>> result = ValidationResult[int](valid=True, converted_value=42)
      >>> bool(result)
      True
      >>> result.with_converted_value("string")
      ValidationResult(valid=True, violations=[], converted_value='string')




   .. py:class:: ValidationResult   :module: 

      Result of type validation with possible conversion.

      This class encapsulates the outcome of a validation operation, including
      whether validation passed, any violations that occurred, and an optional
      converted value that maintains its type through generic constraints.

      The ValidationResult maintains type safety through covariant generics,
      ensuring that type information flows correctly through validation chains
      and transformations. It acts as both a container for validation status
      and a monad-like structure that can be composed and transformed while
      preserving the validation context.

      .. rubric:: Attributes

      valid (bool): Boolean indicating if validation succeeded
      violations (List[TypeViolation]): List of specific type violations encountered
      converted_value (Optional[T]): Optional transformed value that maintains
          its type through generics

      .. rubric:: Examples

      >>> result = ValidationResult[int](valid=True, converted_value=42)
      >>> bool(result)
      True
      >>> result.with_converted_value("string")
      ValidationResult(valid=True, violations=[], converted_value='string')




   .. py:class:: ValidationResult   :module: 

      Result of type validation with possible conversion.

      This class encapsulates the outcome of a validation operation, including
      whether validation passed, any violations that occurred, and an optional
      converted value that maintains its type through generic constraints.

      The ValidationResult maintains type safety through covariant generics,
      ensuring that type information flows correctly through validation chains
      and transformations. It acts as both a container for validation status
      and a monad-like structure that can be composed and transformed while
      preserving the validation context.

      .. rubric:: Attributes

      valid (bool): Boolean indicating if validation succeeded
      violations (List[TypeViolation]): List of specific type violations encountered
      converted_value (Optional[T]): Optional transformed value that maintains
          its type through generics

      .. rubric:: Examples

      >>> result = ValidationResult[int](valid=True, converted_value=42)
      >>> bool(result)
      True
      >>> result.with_converted_value("string")
      ValidationResult(valid=True, violations=[], converted_value='string')




   .. py:class:: ValidatorFactory   :module: 

      Factory class for creating validators dynamically with type safety.

      This class provides static methods for creating various types of validators
      and performing validation operations with comprehensive error reporting.

      The factory pattern enables consistent validator creation with proper
      configuration and composition, supporting both simple and complex
      validation scenarios.



.. py:function:: are_types_compatible(source_type, target_type)

      Check if two types are compatible for conversion or assignment.

      Determines whether values of the source type can generally be
      converted to the target type without errors.

      :param source_type: The source type to check from
      :param target_type: The target type to check compatibility with

      :returns: True if the types are compatible, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> are_types_compatible(int, float)
      True
      >>> are_types_compatible(float, int)
      True
      >>> are_types_compatible(list, tuple)
      True
      >>> are_types_compatible(dict, list)
      False

      .. note::

         This evaluates type compatibility at a general level without
         considering specific value constraints.


.. py:function:: has_attributes(obj, *attributes)

      Check if an object has all the specified attributes.

      :param obj: The object to check
      :param \*attributes: Attribute names to look for

      :returns: True if object has all attributes, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> has_attributes([], "append", "extend")
      True
      >>> has_attributes({}, "update", "missing_attr")
      False
      >>> has_attributes("string", "upper", "lower")
      True
      >>> has_attributes(None, "any_attr")
      False

      .. note:: This checks for attribute existence, not their values or callability.


.. py:function:: is_callable(value)

      Check if a value is callable (function, method, callable object).

      Determines whether an object can be called like a function.

      :param value: The value to check

      :returns: True if the value is callable, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_callable(lambda x: x)
      True
      >>> is_callable(print)
      True
      >>> is_callable("not_callable")
      False
      >>> is_callable(None)
      False

      .. note:: This is a type-safe wrapper around the built-in callable() function.


.. py:function:: is_collection(value)

      Check if a value is a collection (list, tuple, set, dict, etc.).

      Determines whether a value is a collection type that can contain
      multiple elements, excluding strings and bytes which are sequence
      types but not typically treated as collections.

      :param value: The value to check

      :returns: True if the value is a collection, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_collection([1, 2, 3])
      True
      >>> is_collection((1, 2, 3))
      True
      >>> is_collection({"a": 1})
      True
      >>> is_collection("string")  # Strings are not considered collections
      False
      >>> is_collection(42)
      False

      .. note:: Strings and bytes are not considered collections despite being sequences.


.. py:function:: is_compatible_with_type(value, target_type)

      Check if a value can be converted to a target type without errors.

      Determines whether a value can be safely converted to the specified type
      without raising exceptions, allowing for type conversion safety checks.

      :param value: The value to check
      :param target_type: The target type to check compatibility with

      :returns: True if the value can be safely converted, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_compatible_with_type("123", int)
      True
      >>> is_compatible_with_type("hello", int)
      False
      >>> is_compatible_with_type(42, str)
      True
      >>> is_compatible_with_type([1, 2, 3], tuple)
      True

      .. note::

         This performs actual conversion attempts and catches exceptions,
         making it suitable for runtime type compatibility checking.


.. py:function:: is_function(obj)

      Check if an object is a function.

      Determines whether an object is a function (not a method or builtin).

      :param obj: The object to check

      :returns: True if the object is a function, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> def example_function(): pass
      >>> is_function(example_function)
      True
      >>> is_function(len)
      False
      >>> class Example:
      ...     def method(self): pass
      >>> obj = Example()
      >>> is_function(obj.method)  # doctest: +SKIP
      False
      >>> is_function(lambda x: x)
      True

      .. note:: This identifies pure functions, not methods or built-in functions.


.. py:function:: is_in_range(value, min_value, max_value)

      Check if an integer is within a specified range.


.. py:function:: is_instance_of_any(value, types)

      Check if a value is an instance of any of the specified types.

      Determines whether the value is an instance of at least one
      of the types in the provided tuple.

      :param value: The value to check
      :param types: Tuple of types to check against

      :returns: True if value is an instance of any type in types, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_instance_of_any(42, (str, int, float))
      True
      >>> is_instance_of_any("hello", (list, tuple, dict))
      False
      >>> is_instance_of_any(None, (str, int, type(None)))
      True
      >>> is_instance_of_any([], (list, tuple))
      True

      .. note::

         More efficient than multiple isinstance() calls when checking
         against many types.


.. py:function:: is_method(obj)

      Check if an object is a method.

      Determines whether an object is a method bound to a class instance.

      :param obj: The object to check

      :returns: True if the object is a method, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> class Example:
      ...     def method(self): pass
      ...     @classmethod
      ...     def class_method(cls): pass
      ...     @staticmethod
      ...     def static_method(): pass
      >>> obj = Example()
      >>> is_method(obj.method)  # doctest: +SKIP
      True
      >>> is_method(Example.method)  # doctest: +SKIP
      False
      >>> is_method(obj.class_method)  # doctest: +SKIP
      True
      >>> is_method(obj.static_method)  # doctest: +SKIP
      False

      .. note:: This distinguishes between bound methods and regular functions.


.. py:function:: is_non_empty_string(value)

      Verify if a value is a non-empty string.

      Performs type checking and emptiness validation in a single operation
      with maximum efficiency.

      :param value: The value to validate. Can be any Python object.

      :returns: True if the value is a non-empty string, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> is_non_empty_string("hello")
      True
      >>> is_non_empty_string("")
      False
      >>> is_non_empty_string(123)
      False
      >>> is_non_empty_string(None)
      False

      .. note::

         This function uses isinstance for type checking rather than type()
         to properly handle inheritance relationships.


.. py:function:: is_not_empty(value)

      Check if a string is not empty.


.. py:function:: is_numeric(value)

      Check if a value is numeric (int, float, complex, or numeric subclass).

      Determines whether a value is of a numeric type, handling both
      built-in numeric types and numbers.Number subclasses.

      :param value: The value to check

      :returns: True if the value is numeric, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_numeric(42)
      True
      >>> is_numeric(3.14)
      True
      >>> is_numeric(1+2j)
      True
      >>> is_numeric("42")
      False
      >>> is_numeric(None)
      False

      .. note:: This function considers all subclasses of numbers.Number as numeric.


.. py:function:: is_positive(value)

      Check if an integer is positive.


.. py:function:: is_protocol_instance(obj, protocol)

      Check if an object satisfies a Protocol interface.

      Safely determines if an object implements all the methods and attributes
      required by a Protocol, with proper handling of runtime Protocol checking.

      :param obj: Object to check
      :param protocol: Protocol to check against

      :returns: True if the object satisfies the protocol, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> from typing import Protocol
      >>> class SupportsLen(Protocol):
      ...     def __len__(self) -> int: ...
      >>> is_protocol_instance([1, 2, 3], SupportsLen)  # doctest: +SKIP
      True
      >>> is_protocol_instance(42, SupportsLen)  # doctest: +SKIP
      False

      .. note::

         Works with both @runtime_checkable Protocols and regular Protocols.
         For non-runtime-checkable protocols, uses attribute inspection.


.. py:function:: is_subclass_safe(cls, parent)

      Safely check if a class is a subclass of another class.

      Performs an issubclass check that won't raise TypeError if the first
      argument is not a class, unlike the built-in issubclass function.

      :param cls: The potential subclass to check
      :param parent: The parent class(es) to check against

      :returns: True if cls is a subclass of parent, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_subclass_safe(str, object)
      True
      >>> is_subclass_safe("not_a_class", object)
      False
      >>> is_subclass_safe(dict, (list, tuple))
      False
      >>> is_subclass_safe(list, (list, tuple))
      True
      >>> is_subclass_safe(None, object)
      False

      .. note::

         This is a safer version of the built-in issubclass() function that
         won't raise TypeError for non-class objects.


.. py:function:: is_valid_identifier(name)

      Check if a string is a valid Python identifier.

      Validates that a string can be used as a Python variable,
      function, or class name according to Python syntax rules.

      :param name: The string to check

      :returns: True if the string is a valid Python identifier, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> is_valid_identifier("valid_name")
      True
      >>> is_valid_identifier("123invalid")
      False
      >>> is_valid_identifier("also-invalid")
      False
      >>> is_valid_identifier("")
      False
      >>> is_valid_identifier("_private")
      True

      .. note::

         A valid identifier starts with a letter or underscore and contains
         only letters, numbers, and underscores.


.. py:function:: is_valid_length(value, min_length = None, max_length = None)

      Check if a collection has a valid length.

      :param value: The collection to check
      :param min_length: Minimum allowed length (inclusive), or None for no minimum
      :param max_length: Maximum allowed length (inclusive), or None for no maximum

      :returns: True if the length is valid, False otherwise


.. py:data:: DictSchemaT

      Dictionary schema mapping string keys to schema types.

      This represents object structures with string keys mapping to schema-defined values,
      enabling the definition of complex nested objects.

      .. rubric:: Examples

      >>> person_schema: DictSchemaT = {"name": str, "age": int}
      >>> nested_schema: DictSchemaT = {"user": {"id": int, "name": str}}

.. py:data:: SchemaTypeT

      A schema type definition that can recursively represent any data structure.

      This type allows the definition of complex nested structures with full type safety,
      supporting simple types, unions of types, dictionary schemas, and list schemas.

      .. rubric:: Examples

      >>> # Simple type schema
      >>> int_schema: SchemaTypeT = int
      >>>
      >>> # Union type schema (alternative types)
      >>> union_schema: SchemaTypeT = (int, str)
      >>>
      >>> # Dictionary schema (nested object)
      >>> dict_schema: SchemaTypeT = {"name": str, "age": int}
      >>>
      >>> # List schema (array of integers)
      >>> list_schema: SchemaTypeT = [int]

.. py:data:: T

.. py:data:: T

.. py:data:: T

.. py:data:: V

.. py:data:: __version__
      :type:  str
      :value: '0.1.0'


.. py:data:: version
      :value: '0.1.0'


