type_forge.core.base
====================

.. py:module:: type_forge.core.base

.. autoapi-nested-parse::

   Core validation framework with precise type guarantees and recursive refinement.

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



Classes
-------

.. autoapisummary::

   type_forge.core.base.BaseValidator
   type_forge.core.base.TypeForgeBase
   type_forge.core.base.ValidationResult


Module Contents
---------------

.. py:class:: BaseValidator

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


   .. py:method:: validate(value)

      Validate the given value.

      This is the simplified validation interface that returns only a boolean result.
      Subclasses must implement this method to define specific validation logic.

      :param value: The value to validate
      :type value: object

      :returns: True if the value is valid, False otherwise
      :rtype: bool

      :raises NotImplementedError: If the subclass doesn't implement this method



   .. py:method:: validate_with_detail(value)

      Validate the given value with detailed results.

      Provides comprehensive validation information including specific violations
      and an optionally converted value. This implementation relies on the simple
      validate method, but subclasses may override for more detailed reporting.

      This method demonstrates the recursive refinement principle by building
      detailed validation upon the simpler boolean validation. It also implements
      the "Errors as Values" principle by encapsulating failures in the return
      type rather than through exceptions.

      :param value: The value to validate
      :type value: object

      :returns: Detailed validation results with type preservation
      :rtype: ValidationResult[object]

      :raises No direct exceptions, but captures exceptions from validate() and:
      :raises converts them to ValidationResult with appropriate violations:



.. py:class:: TypeForgeBase

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


   .. py:method:: add_validator(validator)

      Add a validator to the type forging process.

      Appends a validator to the internal list of validators that will
      be applied during validation operations. This method implements
      the builder pattern for constructing validation chains.

      :param validator: An instance of a validator
      :type validator: BaseValidator

      :raises TypeError: If validator is not an instance of BaseValidator



   .. py:method:: validate(value)

      Validate a value against all registered validators.

      Performs sequential validation using all registered validators,
      implementing the fail-fast principle by short-circuiting on the first failure
      for efficient validation.

      This method provides a simplified boolean interface to the validation
      process, while maintaining full compatibility with the type registration
      and validation system used throughout TypeForge.

      :param value: The value to validate against the validators
                    in this TypeForgeBase instance
      :type value: object

      :returns: True if the value passes all validators, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> validator = TypeForgeBase()
      >>> validator.add_validator(IntegerValidator())
      >>> validator.validate(42)
      True
      >>> validator.validate("not an integer")
      False



   .. py:method:: validate_with_detail(value)

      Validate with detailed results from all validators.

      Aggregates validation results from all validators, maintaining a comprehensive
      record of any violations that occur while preserving type safety.

      Unlike the simple validate method, this continues to run all validators
      even after failures to collect complete violation information.
      This implementation balances efficiency with completeness, providing
      full failure details for better debugging and error reporting.

      :param value: The value to validate
      :type value: object

      :returns: Aggregated validation details with type preservation
      :rtype: ValidationResult[object]



   .. py:attribute:: validators
      :type:  List[BaseValidator]
      :value: []



.. py:class:: ValidationResult

   Bases: :py:obj:`Generic`\ [\ :py:obj:`type_forge.typing.definitions.T`\ ]


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


   .. py:method:: merge(other)

      Merge another validation result into this one.

      This maintains the type of the original result while incorporating
      the validation status and violations from another result. The converted
      value remains that of the original result, preserving type integrity.

      This method enables recursive composition of validation results,
      allowing complex validation logic to be built from simpler components
      while maintaining full type safety.

      :param other: Another validation result to merge with this one
      :type other: ValidationResult[object]

      :returns: Self with merged validation status and violations
      :rtype: ValidationResult[T]



   .. py:method:: with_converted_value(value)

      Create a new ValidationResult with the same validation status but a different value.

      This method preserves the validation state while transforming the result
      to contain a new value of potentially different type, properly maintaining
      type safety through generics. This enables validation pipelines that
      transform data while maintaining error context.

      The implementation follows the principle of "Pure Core, Effects at Edges"
      by returning a new instance rather than modifying the current one,
      maintaining immutability for better compositional properties.

      :param value: The converted value to set
      :type value: S

      :returns: A new ValidationResult with the same validation status but different value
      :rtype: ValidationResult[S]



   .. py:attribute:: converted_value
      :type:  Optional[type_forge.typing.definitions.T]
      :value: None



   .. py:attribute:: valid
      :type:  bool


   .. py:attribute:: violations
      :type:  List[type_forge.core.exceptions.TypeViolation]
      :value: []



