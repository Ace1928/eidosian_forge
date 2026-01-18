.. py:module:: src.type_forge.typing.mapping

Type Mapping Utilities for the Type Forge System
   ===============================================

   This module contains utilities for type categorization, relationship mapping,
   and descriptive information about types. These functions enable semantic analysis
   of type relationships beyond simple inheritance hierarchies.

   Core functionalities include:
   - Type categorization (atomic, container, protocol, etc.)
   - Finding common supertypes across multiple types
   - Converting between type names and Python type objects
   - Generating human-readable type descriptions
   - Analyzing type structure and relationships

   These utilities serve as the foundation for more complex type operations
   in the Type Forge system, enabling precise type handling with elegant
   error management and comprehensive edge case coverage.


Module Contents
---------------

.. py:function:: describe_type(value)

      Generate a detailed description of a value's type.

      Creates a human-readable description of an object's type,
      including additional information for collections.

      :param value: The value to describe

      :returns: A detailed description of the value's type
      :rtype: str

      .. rubric:: Examples

      >>> describe_type(42)
      'int'
      >>> describe_type([1, 2, 3])
      'list[int] (length: 3)'
      >>> describe_type({'a': 1, 'b': 'text'})
      'dict[str, mixed] (size: 2)'
      >>> describe_type(None)
      'None'

      .. note:: For collections, includes element types and collection size.


.. py:function:: get_common_supertype(types)

      Find the most specific common supertype of multiple types.

      Identifies the closest common ancestor type that all the given
      types inherit from, providing the tightest type bound.

      :param types: List of types to find a common supertype for

      :returns: The common supertype, or None if only object is common
      :rtype: Optional[Type[object]]

      .. rubric:: Examples

      >>> get_common_supertype([int, float])  # doctest: +SKIP
      <class 'numbers.Number'>
      >>> get_common_supertype([list, tuple])  # doctest: +SKIP
      <class 'collections.abc.Sequence'>
      >>> get_common_supertype([str, int]) is object
      True
      >>> get_common_supertype([]) is None
      True

      .. note::

         Returns object if no more specific common supertype exists.
         Returns None for an empty list of types.


.. py:function:: get_python_type_for_name(type_name)

      Get the Python type object corresponding to a type name.

      Maps common type names to their corresponding Python type objects,
      handling both builtin types and common collection types.

      :param type_name: Name of the type as a string

      :returns: The corresponding Python type, or None if not found
      :rtype: Optional[Type[object]]

      .. rubric:: Examples

      >>> get_python_type_for_name("int")
      <class 'int'>
      >>> get_python_type_for_name("str")
      <class 'str'>
      >>> get_python_type_for_name("list")
      <class 'list'>
      >>> get_python_type_for_name("unknown") is None
      True
      >>> get_python_type_for_name("STRING")  # Case insensitive
      <class 'str'>

      .. note::

         Currently handles only common builtin types. For more complex types,
         consider eval() with appropriate safety measures.


.. py:function:: get_type_category(typ)

      Determine the semantic category of a type.

      Categorizes types into meaningful groups based on their structure
      and behavior rather than just their inheritance relationships.

      :param typ: The type to categorize

      :returns: The semantic category of the type
      :rtype: TypeCategory

      .. rubric:: Examples

      >>> get_type_category(int)
      <TypeCategory.ATOMIC: 'atomic'>
      >>> get_type_category(list)
      <TypeCategory.CONTAINER: 'container'>
      >>> get_type_category(dict)
      <TypeCategory.CONTAINER: 'container'>
      >>> get_type_category(Protocol)  # doctest: +SKIP
      <TypeCategory.PROTOCOL: 'protocol'>

      .. note::

         This function uses both inheritance and structural properties
         to determine the category.


.. py:function:: get_type_name(typ)

      Get a user-friendly name for a type object.

      Creates a more readable name for types, handling special cases like
      NoneType and properly formatting generic types.

      :param typ: The type to get a name for

      :returns: A user-friendly name for the type
      :rtype: str

      .. rubric:: Examples

      >>> get_type_name(int)
      'int'
      >>> get_type_name(type(None))
      'None'
      >>> get_type_name(Dict[str, int])  # doctest: +SKIP
      'Dict[str, int]'
      >>> get_type_name(List[str])  # doctest: +SKIP
      'List[str]'
      >>> get_type_name(List)  # doctest: +SKIP
      'List'

      .. note:: This function creates names similar to those used in type annotations.


.. py:data:: MaybeType

.. py:data:: T

.. py:data:: TypeDescription

.. py:data:: TypeName

.. py:data:: TypeObject

.. py:data:: U

.. py:data:: version
      :type:  Final[str]
      :value: '0.1.0'


