type_forge.core.exceptions
==========================

.. py:module:: type_forge.core.exceptions

.. autoapi-nested-parse::

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



Attributes
----------

.. autoapisummary::

   type_forge.core.exceptions.ViolationKindLiteral


Exceptions
----------

.. autoapisummary::

   type_forge.core.exceptions.ConfigurationError
   type_forge.core.exceptions.TypeCreationError
   type_forge.core.exceptions.TypeForgeException
   type_forge.core.exceptions.ValidationError


Classes
-------

.. autoapisummary::

   type_forge.core.exceptions.TypeViolation
   type_forge.core.exceptions.TypeViolationKind


Module Contents
---------------

.. py:exception:: ConfigurationError(message)

   Bases: :py:obj:`TypeForgeException`


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


.. py:exception:: TypeCreationError(message)

   Bases: :py:obj:`TypeForgeException`


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


.. py:exception:: TypeForgeException

   Bases: :py:obj:`Exception`


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


.. py:exception:: ValidationError(message)

   Bases: :py:obj:`TypeForgeException`


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


.. py:class:: TypeViolation

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


   .. py:attribute:: expected
      :type:  str


   .. py:attribute:: found
      :type:  str


   .. py:attribute:: kind
      :type:  TypeViolationKind


   .. py:attribute:: path
      :type:  str


.. py:class:: TypeViolationKind(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


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


   .. py:attribute:: CONVERSION_ERROR
      :value: 'conversion_error'



   .. py:attribute:: INVALID_VALUE
      :value: 'invalid_value'



   .. py:attribute:: MISSING_KEY
      :value: 'missing_key'



   .. py:attribute:: SCHEMA_MISMATCH
      :value: 'schema_mismatch'



   .. py:attribute:: WRONG_TYPE
      :value: 'wrong_type'



.. py:data:: ViolationKindLiteral

