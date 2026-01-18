.. py:module:: src.type_forge.core.exceptions

Exception hierarchy and type violation tracking for TypeForge.

   This module provides a comprehensive exception system for the TypeForge library,
   enabling precise error reporting, structured violation tracking, and clear
   error hierarchies that maintain the principle of informative failure.

   Classes:
       TypeForgeException: Base exception for all TypeForge errors.
       ValidationError: Raised when validation constraints aren't met.
       TypeCreationError: Raised when type construction fails.
       ConfigurationError: Raised when configuration is invalid.
       TypeViolationKind: Enumeration of violation categories.
       TypeViolation: Immutable record of a type violation with path tracking.


Module Contents
---------------


   .. py:class:: ConfigurationError(message)   :module: 

      Exception raised for configuration-related errors.

      Raised when TypeForge is configured with invalid, incompatible,
      or missing configuration parameters.

      :param message: Descriptive error message explaining the configuration issue.

      .. rubric:: Examples

      ```python
      raise ConfigurationError("Invalid serialization format specified")
      ```

      Initialize with error message.

      :param message: Detailed description of the configuration error.




   .. py:class:: TypeCreationError(message)   :module: 

      Exception raised for errors during type creation.

      Raised when attempting to create a type definition fails due to
      invalid parameters, conflicting constraints, or other type
      construction issues.

      :param message: Descriptive error message explaining the type creation failure.

      .. rubric:: Examples

      ```python
      raise TypeCreationError("Cannot create recursive type without base case")
      ```

      Initialize with error message.

      :param message: Detailed description of the type creation error.




   .. py:class:: TypeForgeException   :module: 

      Base class for all exceptions raised by the TypeForge module.

      All exceptions in this library inherit from this class, enabling
      targeted exception handling for TypeForge-specific errors.

      .. rubric:: Examples

      ```python
      try:
          # Some TypeForge operation
          pass
      except TypeForgeException as e:
          # Handle any TypeForge-related error
          pass
      ```

      Initialize self.  See help(type(self)) for accurate signature.




   .. py:class:: ValidationError(message)   :module: 

      Exception raised for data validation errors.

      Raised when data fails to meet validation constraints defined
      in a type schema or validation rule.

      :param message: Descriptive error message explaining the validation failure.

      .. rubric:: Examples

      ```python
      raise ValidationError("Age must be greater than 0")
      ```

      Initialize with error message.

      :param message: Detailed description of the validation error.




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



.. py:data:: ViolationKindLiteral

