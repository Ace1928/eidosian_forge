type_forge.utils.string_format
==============================

.. py:module:: type_forge.utils.string_format


Functions
---------

.. autoapisummary::

   type_forge.utils.string_format.camel_to_snake
   type_forge.utils.string_format.format_message
   type_forge.utils.string_format.format_type_name
   type_forge.utils.string_format.pluralize
   type_forge.utils.string_format.snake_to_camel
   type_forge.utils.string_format.standardize_typename


Module Contents
---------------

.. py:function:: camel_to_snake(name)

   Convert a camelCase or PascalCase string to snake_case.

   Transforms identifiers from camelCase or PascalCase convention to
   snake_case for consistent naming across the system.

   :param name: The camelCase or PascalCase string to convert
   :type name: str

   :returns: The equivalent snake_case string
   :rtype: str

   .. rubric:: Examples

   >>> camel_to_snake("camelCase")
   'camel_case'
   >>> camel_to_snake("PascalCase")
   'pascal_case'
   >>> camel_to_snake("HTTPResponse")
   'http_response'
   >>> camel_to_snake("")
   ''

   .. note:: Handles special cases with consecutive uppercase letters correctly.


.. py:function:: format_message(subject, value)

   Format a message by combining a string subject with an integer value.

   Creates a precisely formatted string combining the subject and value
   in a grammatically coherent structure.

   :param subject: The subject of the message. Must be a valid string.
   :type subject: str
   :param value: The numerical value associated with the subject. Must be an integer.
   :type value: int

   :returns: A formatted message combining the subject and value.
   :rtype: str

   .. rubric:: Examples

   >>> format_message("Temperature", 72)
   'Temperature has a value of 72.'

   .. note:: This function performs no validation on the inputs beyond typing.


.. py:function:: format_type_name(typ)

   Format a type into a clean, readable string representation.

   Converts Python type objects into standardized string representations
   for consistent display and logging. Handles generics, unions, and nested types.

   :param typ: The type to format into a string
   :type typ: Type[object]

   :returns: A standardized string representation of the type
   :rtype: str

   .. rubric:: Examples

   >>> format_type_name(int)
   'int'
   >>> format_type_name(List[int])  # doctest: +SKIP
   'List[int]'
   >>> format_type_name(Dict[str, Optional[int]])  # doctest: +SKIP
   'Dict[str, Optional[int]]'

   .. note:: This function handles generic types, union types, and nested types.


.. py:function:: pluralize(word)

   Convert a singular word to its plural form using simple English rules.

   Applies common English pluralization rules to transform singular nouns
   to their plural forms.

   :param word: The singular word to pluralize
   :type word: str

   :returns: The pluralized form of the word
   :rtype: str

   .. rubric:: Examples

   >>> pluralize("cat")
   'cats'
   >>> pluralize("class")
   'classes'
   >>> pluralize("study")
   'studies'
   >>> pluralize("box")
   'boxes'
   >>> pluralize("child")  # Special cases are not handled
   'childs'

   .. note:: Handles common English pluralization rules but not irregular nouns.


.. py:function:: snake_to_camel(name)

   Convert a snake_case string to camelCase.

   Transforms identifiers from snake_case convention to camelCase
   for consistent naming across the system.

   :param name: The snake_case string to convert
   :type name: str

   :returns: The equivalent camelCase string
   :rtype: str

   .. rubric:: Examples

   >>> snake_to_camel("snake_case")
   'snakeCase'
   >>> snake_to_camel("http_response")
   'httpResponse'
   >>> snake_to_camel("__private_var")
   '__privateVar'
   >>> snake_to_camel("")
   ''

   .. note:: Preserves leading underscores for private/protected identifiers.


.. py:function:: standardize_typename(name)

   Standardize a type name for consistent representation.

   Converts various representations of type names to a standardized format,
   handling common variations and aliases.

   :param name: The type name to standardize
   :type name: str

   :returns: The standardized type name
   :rtype: str

   .. rubric:: Examples

   >>> standardize_typename("int")
   'int'
   >>> standardize_typename("integer")
   'int'
   >>> standardize_typename("str")
   'str'
   >>> standardize_typename("string")
   'str'
   >>> standardize_typename("bool")
   'bool'
   >>> standardize_typename("boolean")
   'bool'

   .. note:: Normalizes common type name variations to their Python equivalents.


