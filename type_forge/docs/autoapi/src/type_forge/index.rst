.. py:module:: src.type_forge

Type Forge: A module for dynamic type validation and transformation with recursive precision.

   This module serves as the entry point for the Type Forge framework,
   providing a unified interface for dynamic type creation, validation,
   and transformation with structural integrity and recursive precision.

   The TypeForge framework enables:
       - Dynamic type creation with strict validation guarantees
       - Recursive schema validation with elegant error handling
       - Type conversion with compile-time and runtime safety
       - Structural integrity verification through composable validators

   Modules:
       core: Contains base classes, protocols, and exceptions for type validation
       forge: Implements the main TypeForge class and associated functionality
       validators: Provides specialized validators for comprehensive type checking
       typing: Type definitions, protocols, and type-related utilities
       utils: General utility functions for type manipulation and formatting

   Classes:
       TypeForge: Main entry point for dynamic type creation and validation
       BasicValidator: Foundation validator for simple type validation
       CompositeValidator: Combines multiple validators for complex validation
       ValidationResult: Immutable result of a validation operation
       TypeViolation: Precisely describes a type violation with context
       TypeViolationKind: Enumeration of possible validation failure types
       ValidatorFactory: Creates specialized validators for diverse scenarios

   Functions:
       format_validation_error(violation: TypeViolation) -> str:
           Formats type violations as human-readable error messages
       deduplicate_violations(violations: Sequence[TypeViolation]) -> List[TypeViolation]:
           Removes duplicate violations while preserving order and context
       is_valid_type_descriptor(value: Any) -> bool:
           Determines if a value can serve as a valid type descriptor

   Typical usage example:
       >>> from type_forge import TypeForge
       >>> forge = TypeForge()
       >>> UserSchema = forge.create({
       ...     "name": str,
       ...     "age": int,
       ...     "email": forge.Email()
       ... })
       >>> user = UserSchema(name="Alice", age=30, email="alice@example.com")


Package Contents
----------------


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



