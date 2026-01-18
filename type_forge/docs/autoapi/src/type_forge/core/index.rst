.. py:module:: src.type_forge.core

Core module for the TypeForge type validation system.

   This module provides the essential components for runtime type validation,
   dynamic type creation, and structural type checking. It forms the backbone
   of the TypeForge framework, enabling precise control over data structures
   with minimal runtime overhead.

   The components in this module follow composition patterns with strict
   typing to ensure type safety throughout the validation pipeline. Each
   component has a single responsibility and communicates through well-defined
   interfaces.

   Architecture:
       TypeForgeBase: The root class establishing the type identity system
       BaseValidator: Validation interface with pure functional core
       ValidationResult: Immutable result structure with strict guarantees
       Exceptions: Specialized error hierarchy for precise error handling

   Exports:
       BaseValidator: Abstract base class defining the validator interface
           with validate() and is_valid() methods
       TypeForgeBase: Root class for all TypeForge objects providing common
           functionality and type identity
       ValidationResult: Immutable container for validation outcomes with
           status, value, and violation details
       ValidationError: Exception raised when validation fails with
           detailed violation information
       TypeForgeException: Base exception class for all TypeForge errors
           to enable specific error handling
       TypeCreationError: Exception raised when type construction fails
           due to invalid parameters or configuration
       ConfigurationError: Exception for invalid TypeForge configuration
           typically raised during initialization
       TypeViolation: Detailed information about a specific type violation
           including path, value, and violation kind
       TypeViolationKind: Enumeration of possible violation categories
           (e.g., type mismatch, missing field, extra field)

   .. admonition:: Example

      >>> from type_forge.core import ValidationResult, TypeForgeBase, TypeViolation, TypeViolationKind
      >>>
      >>> # Create a validation result for a successful validation
      >>> result = ValidationResult(valid=True, value=42, violations=[])
      >>> assert isinstance(result, TypeForgeBase)
      >>>
      >>> # Example with validation failure
      >>> violations = [TypeViolation(
      ...     path="age",
      ...     expected="int",
      ...     received="str",
      ...     value="twenty",
      ...     kind=TypeViolationKind.TYPE_MISMATCH
      ... )]
      >>> failed_result = ValidationResult(valid=False, value=None, violations=violations)
      >>> assert not failed_result.valid
      >>>
      >>> # Accessing violation details
      >>> if not failed_result.valid:
      ...     for violation in failed_result.violations:
      ...         print(f"Error at {violation.path}: Expected {violation.expected, got {violation.received}")

   .. admonition:: Notes

      All components maintain strict immutability to ensure thread safety and
      prevent side-effects during validation operations.

   .. seealso::

      type_forge.validators: Higher-level validation components
      type_forge.types: Type definition and construction utilities


Package Contents
----------------


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




   .. py:class:: TypeForgeBase   :module: 

      Base class for the type forging process.

      Orchestrates validation through multiple validators, providing both
      simple boolean validation and detailed validation results with proper
      type preservation.

      The TypeForgeBase implements the composition pattern, allowing multiple
      validators to be combined while maintaining a consistent interface
      and preserving type information throughout the validation process.
      This embodies the Eidosian principle of "Fractal Coherence" where
      complex validation logic emerges from simpler components in a
      consistent manner.

      .. rubric:: Attributes

      validators (List[BaseValidator]): List of validators to apply during validation

      Initialize with an empty validators list.

      Creates a new TypeForgeBase instance with no validators.
      Validators must be added using the add_validator method.
      This follows the "Data Before Behavior" principle by establishing
      the core data structure before defining operations on it.




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



