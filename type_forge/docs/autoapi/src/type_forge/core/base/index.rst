.. py:module:: src.type_forge.core.base

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


Module Contents
---------------


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



