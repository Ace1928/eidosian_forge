type_forge.validators.factory
=============================

.. py:module:: type_forge.validators.factory


Attributes
----------

.. autoapisummary::

   type_forge.validators.factory.V


Classes
-------

.. autoapisummary::

   type_forge.validators.factory.ValidatorFactory


Module Contents
---------------

.. py:class:: ValidatorFactory

   Factory class for creating validators dynamically with type safety.

   This class provides static methods for creating various types of validators
   and performing validation operations with comprehensive error reporting.

   The factory pattern enables consistent validator creation with proper
   configuration and composition, supporting both simple and complex
   validation scenarios.


   .. py:method:: create_validator(validator_type, **kwargs)
      :staticmethod:


      Creates a validator based on the specified type.

      :param validator_type: The type of validator to create.
                             Must be one of the supported validator types ('basic', 'composite').
      :param \*\*kwargs: Additional parameters for the validator.
                         Passed directly to the validator constructor.

      :returns: An instance of the requested validator.

      :raises ValueError: If the validator type is unknown.

      .. rubric:: Examples

      >>> validator = ValidatorFactory.create_validator('basic')
      >>> validator(42)
      True



   .. py:method:: validate_dict(value, schema, path = '$', convert = False, require_all_keys = True)
      :staticmethod:


      Validate that a dictionary conforms to a schema.

      :param value: Dictionary to validate
      :param schema: Schema defining expected types for keys
      :param path: Current path for error reporting
      :param convert: Whether to attempt type conversion
      :param require_all_keys: Whether all schema keys must be present

      :returns: ValidationResult with validation status and converted dict if applicable

      .. rubric:: Examples

      >>> schema = {"name": str, "age": int}
      >>> data = {"name": "Alice", "age": 30}
      >>> result = ValidatorFactory.validate_dict(data, schema)
      >>> result.valid
      True

      >>> data = {"name": "Bob"}  # Missing 'age'
      >>> result = ValidatorFactory.validate_dict(data, schema)
      >>> result.valid
      False



   .. py:method:: validate_recursive(value, schema, path = '$', convert = False)
      :staticmethod:


      Recursively validate a value against a schema of arbitrary complexity.

      :param value: Value to validate
      :param schema: Schema to validate against (dict, list/sequence, or type)
      :param path: Current path for error reporting
      :param convert: Whether to attempt type conversion

      :returns: ValidationResult with validation status and details

      .. rubric:: Examples

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



   .. py:method:: validate_type(value, expected_type, path = '$', convert = False)
      :staticmethod:


      Validate that a value matches the expected type recursively.

      :param value: The value to check
      :param expected_type: Type or types to check against
      :param path: Current path in the validation (for error reporting)
      :param convert: Whether to attempt conversion to the expected type

      :returns: ValidationResult with validation status and details

      .. rubric:: Examples

      >>> result = ValidatorFactory.validate_type(42, int)
      >>> result.valid
      True

      >>> result = ValidatorFactory.validate_type("42", int, convert=True)
      >>> result.valid
      True
      >>> result.converted_value
      42



.. py:data:: V

