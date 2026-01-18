.. py:module:: src.type_forge.typing.naming

Type Naming Utilities for the Type Forge System
   ===============================================

   This module provides utilities for obtaining standardized names and
   representations of Python types. It ensures consistent naming conventions
   and human-readable type descriptions across the Type Forge system.

   Core functionalities include:
   - Generating standardized type names for both built-in and custom types
   - Handling complex generic types with nested parameters
   - Identifying primitive types for special treatment
   - Providing clean string representations of type hierarchies
   - Converting between different type representation formats
   - Analyzing type structures for naming consistency

   These utilities form the foundation for consistent type naming and
   presentation throughout the Type Forge ecosystem, supporting both
   programmatic type operations and human-readable documentation.


Module Contents
---------------


   .. py:class:: TypeProtocol   :module: 

      Protocol defining the interface for type objects.



.. py:function:: are_types_compatible(source_type, target_type)

      Check if two types are compatible.

      Determines if a value of source_type can be used where target_type is expected.

      :param source_type: The source type to check
      :param target_type: The target type to check against

      :returns: True if the types are compatible, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> are_types_compatible(int, float)
      True
      >>> from typing import List
      >>> are_types_compatible(List[int], List[float])
      False
      >>> from typing import Any
      >>> are_types_compatible(str, Any)
      True

      .. note:: Compatibility checks consider subclass relationships and type hierarchies.


.. py:function:: describe_type(typ)

      Get a detailed description of a type.

      Returns a dictionary with comprehensive type information.

      :param typ: The type to describe

      :returns: A dictionary containing type details
      :rtype: Dict[str, object]

      .. rubric:: Examples

      >>> details = describe_type(int)
      >>> details['name']
      'int'
      >>> details['category']
      'primitive'

      .. note:: Provides deep type structure analysis for complex inspections.


.. py:function:: format_type_annotation(typ, for_docstring = False)

      Format a type for use in annotations or docstrings.

      Generates properly formatted type strings for different contexts.

      :param typ: The type to format
      :param for_docstring: Whether the type is being formatted for a docstring

      :returns: The formatted type annotation
      :rtype: str

      .. rubric:: Examples

      >>> from typing import List, Optional
      >>> format_type_annotation(List[int])
      'List[int]'
      >>> format_type_annotation(Optional[str], for_docstring=True)
      'str or None'

      .. note:: When formatting for docstrings, uses more human-readable forms.


.. py:function:: get_fully_qualified_name(typ)

      Get the fully qualified name of a type.

      Returns the complete module path and type name.

      :param typ: The type to get the name for

      :returns: The fully qualified name of the type
      :rtype: str

      .. rubric:: Examples

      >>> get_fully_qualified_name(int)
      'builtins.int'
      >>> import collections
      >>> get_fully_qualified_name(collections.defaultdict)  # doctest: +SKIP
      'collections.defaultdict'

      .. note:: Useful for serialization and type lookup across module boundaries.


.. py:function:: get_generic_args(typ)

      Get the type arguments of a generic type.

      For generic types like List[int], returns the inner types (int).

      :param typ: The generic type to inspect

      :returns: The type arguments of the generic type
      :rtype: Tuple[Any, ...]

      .. rubric:: Examples

      >>> from typing import List, Dict
      >>> get_generic_args(List[int])
      (int,)
      >>> get_generic_args(Dict[str, int])
      (str, int)
      >>> get_generic_args(int)
      ()

      .. note:: Returns an empty tuple for non-generic types.


.. py:function:: get_type_category(typ)

      Get the category of a type.

      Categorizes a type into one of: primitive, container, composite, callable, or special.

      :param typ: The type to categorize

      :returns: The category of the type
      :rtype: TypeCategoryLiteral

      .. rubric:: Examples

      >>> get_type_category(int)
      'primitive'
      >>> get_type_category(list)
      'container'
      >>> get_type_category(Union[int, str])
      'composite'
      >>> get_type_category(Callable[[int], str])
      'callable'

      .. note:: Categories help determine how types should be processed or displayed.


.. py:function:: get_type_name(typ)

      Get the name of a type.

      For simple types, returns the type name.
      For generic types, includes the parameter types.

      :param typ: The type to get the name for

      :returns: The name of the type
      :rtype: str

      .. rubric:: Examples

      >>> get_type_name(int)
      'int'
      >>> from typing import List
      >>> get_type_name(List[int])
      'List[int]'


.. py:function:: is_container_type(typ)

      Check if a type is a container.

      Container types include list, dict, set, tuple, etc.

      :param typ: The type to check

      :returns: True if the type is a container, False otherwise
      :rtype: bool


.. py:function:: is_generic_type(typ)

      Check if a type is generic.

      Generic types include List[T], Dict[K, V], etc.

      :param typ: The type to check

      :returns: True if the type is generic, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> from typing import List
      >>> is_generic_type(List[int])
      True
      >>> is_generic_type(int)
      False


.. py:function:: is_optional_type(typ)

      Check if a type is Optional[T].

      Detects if a type is Union[T, None] or Optional[T].

      :param typ: The type to check

      :returns: True if the type is Optional, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> from typing import Optional
      >>> is_optional_type(Optional[int])
      True
      >>> is_optional_type(int)
      False
      >>> is_optional_type(Union[str, None])
      True

      .. note:: Optional[T] is equivalent to Union[T, None].


.. py:function:: is_primitive_type(typ)

      Check if a type is primitive.

      Primitive types are basic types like int, str, float, bool, etc.

      :param typ: The type to check

      :returns: True if the type is primitive, False otherwise
      :rtype: bool


.. py:function:: normalize_type(typ)

      Normalize a type representation.

      Converts type aliases to their canonical form for consistent representation.

      :param typ: The type to normalize

      :returns: The normalized type
      :rtype: Any

      .. rubric:: Examples

      >>> from typing import List, Sequence
      >>> normalize_type(List[int]) == list[int]  # Python 3.9+ syntax
      True

      .. note:: Normalization ensures consistent type representations across the system.


.. py:data:: CALLABLE_CATEGORY
      :type:  TypeCategoryLiteral
      :value: 'callable'


.. py:data:: COMPOSITE_CATEGORY
      :type:  TypeCategoryLiteral
      :value: 'composite'


.. py:data:: CONTAINER_CATEGORY
      :type:  TypeCategoryLiteral
      :value: 'container'


.. py:data:: PRIMITIVE_CATEGORY
      :type:  TypeCategoryLiteral
      :value: 'primitive'


.. py:data:: SPECIAL_CATEGORY
      :type:  TypeCategoryLiteral
      :value: 'special'


.. py:data:: T

.. py:data:: TypeCategoryLiteral

.. py:data:: U

.. py:data:: V

.. py:data:: version
      :type:  Final[str]
      :value: '0.1.0'


