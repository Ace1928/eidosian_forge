type_forge.typing.standardization
=================================

.. py:module:: type_forge.typing.standardization

.. autoapi-nested-parse::

   Type Standardization Utilities for the Type Forge System
   ========================================================

   This module provides utilities for standardizing, analyzing, and transforming
   type relationships. It ensures consistent type handling across the Type Forge
   ecosystem with precise type categorization and normalization capabilities.

   Core functionalities include:
   - Standardizing type names to consistent PEP 484 conventions
   - Finding common supertypes across multiple types
   - Identifying generic, abstract, and specialized types
   - Retrieving complete type hierarchies via Method Resolution Order (MRO)
   - Deduplicating redundant types from collections

   These utilities enable robust type analysis and transformation operations,
   supporting both static type checking and runtime type verification with
   comprehensive edge case handling.



Attributes
----------

.. autoapisummary::

   type_forge.typing.standardization.version


Functions
---------

.. autoapisummary::

   type_forge.typing.standardization.deduplicate_types
   type_forge.typing.standardization.get_common_supertype
   type_forge.typing.standardization.get_type_hierarchy
   type_forge.typing.standardization.is_abstract_type
   type_forge.typing.standardization.is_generic_type
   type_forge.typing.standardization.standardize_type_name


Module Contents
---------------

.. py:function:: deduplicate_types(types)

   Remove redundant types from a sequence of types.

   Removes types that are subtypes of other types in the sequence,
   keeping only the most specific types necessary.

   :param types: A sequence of types to deduplicate

   :returns: A deduplicated list of types
   :rtype: List[Type[object]]

   .. rubric:: Examples

   >>> deduplicate_types([int, object, float, numbers.Number])  # doctest: +SKIP
   [<class 'int'>, <class 'float'>]
   >>> deduplicate_types([list, Collection, Sequence])  # doctest: +SKIP
   [<class 'list'>]

   .. note:: Useful for simplifying Union types and type annotations.


.. py:function:: get_common_supertype(types)

   Find the most specific common supertype of a sequence of types.

   Determines the most specific type that all provided types inherit from,
   which is useful for type unification.

   :param types: A sequence of types to find the common supertype for

   :returns:

             The common supertype, or None if no common type exists
                                    besides 'object'
   :rtype: Optional[Type[object]]

   .. rubric:: Examples

   >>> get_common_supertype([int, float])  # doctest: +SKIP
   <class 'numbers.Number'>
   >>> get_common_supertype([list, tuple])  # doctest: +SKIP
   <class 'collections.abc.Sequence'>
   >>> get_common_supertype([int, str])  # doctest: +SKIP
   <class 'object'>

   .. note::

      This function traverses inheritance hierarchies to find meaningful
      common supertypes beyond just the 'object' class.


.. py:function:: get_type_hierarchy(typ)

   Get the complete inheritance hierarchy of a type.

   Returns the Method Resolution Order (MRO) of a type, which
   represents its inheritance hierarchy.

   :param typ: The type to get the hierarchy for

   :returns:

             The inheritance hierarchy of the type, from
                                most specific to most general
   :rtype: List[Type[object]]

   .. rubric:: Examples

   >>> bool_hierarchy = get_type_hierarchy(bool)
   >>> bool_hierarchy  # doctest: +SKIP
   [<class 'bool'>, <class 'int'>, <class 'object'>]

   .. note:: Useful for understanding type relationships and finding common types.


.. py:function:: is_abstract_type(typ)

   Check if a type is abstract (cannot be instantiated directly).

   Determines whether a type is an abstract base class or interface
   that cannot be instantiated directly.

   :param typ: The type to check

   :returns: True if the type is abstract, False otherwise
   :rtype: bool

   .. rubric:: Examples

   >>> from abc import ABC, abstractmethod
   >>> class AbstractExample(ABC):
   ...     @abstractmethod
   ...     def method(self): pass
   >>> is_abstract_type(AbstractExample)  # doctest: +SKIP
   True
   >>> is_abstract_type(int)
   False

   .. note:: Identifies types that are meant to be inherited from rather than instantiated.


.. py:function:: is_generic_type(typ)

   Check if a type is a generic type.

   Determines whether a type is a parameterized generic type
   like List[int] rather than a concrete type like list.

   :param typ: The type to check

   :returns: True if the type is a generic type, False otherwise
   :rtype: bool

   .. rubric:: Examples

   >>> is_generic_type(List[int])  # doctest: +SKIP
   True
   >>> is_generic_type(list)
   False
   >>> is_generic_type(Dict[str, int])  # doctest: +SKIP
   True

   .. note:: Useful for handling generic types specially in type systems.


.. py:function:: standardize_type_name(name)

   Standardize a type name to a consistent format.

   Converts various styles of type names to a consistent format
   following PEP 484 conventions.

   :param name: The type name to standardize

   :returns: The standardized type name
   :rtype: str

   .. rubric:: Examples

   >>> standardize_type_name("int")
   'int'
   >>> standardize_type_name("list[int]")
   'List[int]'
   >>> standardize_type_name("Dict[str, list]")
   'Dict[str, List]'

   .. note:: Useful for ensuring consistent type naming across a codebase.


.. py:data:: version
   :value: '0.1.0'


