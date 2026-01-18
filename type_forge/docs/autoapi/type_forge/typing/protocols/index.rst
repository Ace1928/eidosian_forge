type_forge.typing.protocols
===========================

.. py:module:: type_forge.typing.protocols

.. autoapi-nested-parse::

   Type Protocols Module
   ====================

   This module defines protocol classes that specify interfaces for different
   type operations, conversions, and validations in the type forge system.

   These protocols enable static typing and runtime interface verification for
   objects that implement specific behaviors without requiring inheritance.



Attributes
----------

.. autoapisummary::

   type_forge.typing.protocols.version


Classes
-------

.. autoapisummary::

   type_forge.typing.protocols.CompositeValidator
   type_forge.typing.protocols.SupportsBoolConversion
   type_forge.typing.protocols.SupportsComparison
   type_forge.typing.protocols.SupportsEquality
   type_forge.typing.protocols.SupportsFloat
   type_forge.typing.protocols.SupportsFloatConversion
   type_forge.typing.protocols.SupportsGetAttr
   type_forge.typing.protocols.SupportsGetItem
   type_forge.typing.protocols.SupportsInt
   type_forge.typing.protocols.SupportsIntConversion
   type_forge.typing.protocols.SupportsIteration
   type_forge.typing.protocols.SupportsLen
   type_forge.typing.protocols.SupportsLength
   type_forge.typing.protocols.SupportsMapping
   type_forge.typing.protocols.SupportsStrConversion
   type_forge.typing.protocols.SupportsTypeCheck
   type_forge.typing.protocols.TypeConverter
   type_forge.typing.protocols.TypeDeduplicator
   type_forge.typing.protocols.TypeFactory
   type_forge.typing.protocols.TypeForge
   type_forge.typing.protocols.TypeInfo
   type_forge.typing.protocols.TypeNormalizer
   type_forge.typing.protocols.TypeRegistry
   type_forge.typing.protocols.TypeStandardizer
   type_forge.typing.protocols.TypedConverter
   type_forge.typing.protocols.Validator


Module Contents
---------------

.. py:class:: CompositeValidator

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T`\ ]


   Protocol for validators that combine multiple validation rules.

   Type Args:
       T: The type of value being validated.
           For proper variance handling, use this protocol with specific types.

   .. rubric:: Examples

   >>> class AndValidator:
   ...     def __init__(self) -> None:
   ...         self.validators = []
   ...     def add_validator(self, validator):
   ...         self.validators.append(validator)
   ...     def validate(self, value):
   ...         return all(v.validate(value) for v in self.validators)


   .. py:method:: add_validator(validator)

      Adds a validator to the composite validator.

      :param validator: The validator to add.



   .. py:method:: validate(value)

      Validates the given value using all added validators.

      :param value: The value to validate.

      :returns: True if the value passes all validators, False otherwise.
      :rtype: bool



.. py:class:: SupportsBoolConversion

   Bases: :py:obj:`Protocol`


   Protocol for types that can be converted to bool.

   This protocol defines the interface for objects that support
   conversion to boolean values through the __bool__ method.

   .. rubric:: Examples

   >>> class CustomBoolean:
   ...     def __init__(self, value: bool) -> None:
   ...         self.value = value
   ...     def __bool__(self) -> bool:
   ...         return self.value
   >>> bool(CustomBoolean(True))  # True


.. py:class:: SupportsComparison

   Bases: :py:obj:`Protocol`


   Protocol for types that support comparison operations.

   This protocol defines the interface for objects that can be compared
   using standard comparison operators.

   .. rubric:: Examples

   >>> class ComparableValue:
   ...     def __init__(self, value: int) -> None:
   ...         self.value = value
   ...     def __lt__(self, other: object) -> bool:
   ...         if isinstance(other, ComparableValue):
   ...             return self.value < other.value
   ...         return NotImplemented
   >>> ComparableValue(1) < ComparableValue(2)  # True


.. py:class:: SupportsEquality

   Bases: :py:obj:`Protocol`


   Protocol for objects that support equality operations.


.. py:class:: SupportsFloat

   Bases: :py:obj:`Protocol`


   Protocol for objects that support conversion to float.


.. py:class:: SupportsFloatConversion

   Bases: :py:obj:`Protocol`


   Protocol for types that can be converted to float.

   This protocol defines the interface for objects that support
   conversion to floating-point values through the __float__ method.

   .. rubric:: Examples

   >>> class CustomFloat:
   ...     def __init__(self, value: float) -> None:
   ...         self.value = value
   ...     def __float__(self) -> float:
   ...         return self.value
   >>> float(CustomFloat(3.14))  # 3.14


.. py:class:: SupportsGetAttr

   Bases: :py:obj:`Protocol`


   Protocol for types that support attribute access.

   This protocol defines the interface for objects that implement
   custom attribute access through the __getattr__ method.

   .. rubric:: Examples

   >>> class CustomObject:
   ...     def __getattr__(self, name: str) -> int:
   ...         return len(name)
   >>> obj = CustomObject()
   >>> obj.attribute  # 9


.. py:class:: SupportsGetItem

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.K_contra`\ , :py:obj:`type_forge.typing.variables.V_co`\ ]


   Protocol for types that support item access with square brackets.

   This protocol defines the interface for objects that can be accessed
   using the subscription notation (obj[key]).

   Type Args:
       K_contra: The key type (contravariant). This allows a container that
                accepts a supertype to be used where a container that accepts
                a subtype is expected.
       V_co: The value type (covariant). This allows a container that returns
             a subtype to be used where a container that returns a supertype
             is expected.

   .. rubric:: Examples

   >>> class CustomContainer:
   ...     def __init__(self, data: dict[str, int]) -> None:
   ...         self.data = data
   ...     def __getitem__(self, key: str) -> int:
   ...         return self.data[key]
   >>> container = CustomContainer({"one": 1})
   >>> container["one"]  # 1


.. py:class:: SupportsInt

   Bases: :py:obj:`Protocol`


   Protocol for objects that support conversion to int.


.. py:class:: SupportsIntConversion

   Bases: :py:obj:`Protocol`


   Protocol for types that can be converted to int.

   This protocol defines the interface for objects that support
   conversion to integer values through the __int__ method.

   .. rubric:: Examples

   >>> class CustomInteger:
   ...     def __init__(self, value: int) -> None:
   ...         self.value = value
   ...     def __int__(self) -> int:
   ...         return self.value
   >>> isinstance(CustomInteger(42), SupportsIntConversion)  # True at runtime


.. py:class:: SupportsIteration

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T_co`\ ]


   Protocol for types that support iteration.

   This protocol defines the interface for objects that can be iterated over
   using a for loop or with the iter() function.

   Type Args:
       T_co: The type of items yielded by the iterator (covariant).
             This allows an iterator of a subtype to be used where an
             iterator of a supertype is expected.

   .. rubric:: Examples

   >>> class CustomIterable:
   ...     def __init__(self, values: list[int]) -> None:
   ...         self.values = values
   ...     def __iter__(self) -> Iterator[int]:
   ...         return iter(self.values)
   >>> list(CustomIterable([1, 2, 3]))  # [1, 2, 3]


.. py:class:: SupportsLen

   Bases: :py:obj:`Protocol`


   Protocol for objects that support length operations.


.. py:class:: SupportsLength

   Bases: :py:obj:`Protocol`


   Protocol for types that support the len() function.

   This protocol defines the interface for objects that can report
   their length through the __len__ method.

   .. rubric:: Examples

   >>> class CustomSized:
   ...     def __init__(self, size: int) -> None:
   ...         self.size = size
   ...     def __len__(self) -> int:
   ...         return self.size
   >>> len(CustomSized(5))  # 5


.. py:class:: SupportsMapping

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.K_co`\ , :py:obj:`type_forge.typing.variables.V`\ ]


   Protocol for types that support dictionary-like mapping operations.

   This protocol defines the interface for objects that implement mapping
   behavior with key-value pairs.

   Type Args:
       K: The key type.
       V: The value type.

   .. rubric:: Examples

   >>> class CustomMapping:
   ...     def __init__(self) -> None:
   ...         self._data: dict[str, int] = {}
   ...     def __getitem__(self, key: str) -> int:
   ...         return self._data[key]
   ...     def __setitem__(self, key: str, value: int) -> None:
   ...         self._data[key] = value
   ...     def __contains__(self, key: object) -> bool:
   ...         return key in self._data
   >>> mapping = CustomMapping()
   >>> mapping["key"] = 1
   >>> "key" in mapping  # True
   >>> mapping["key"]  # 1


.. py:class:: SupportsStrConversion

   Bases: :py:obj:`Protocol`


   Protocol for types that can be converted to str.

   This protocol defines the interface for objects that support
   conversion to string values through the __str__ method.

   .. rubric:: Examples

   >>> class CustomString:
   ...     def __init__(self, value: str) -> None:
   ...         self.value = value
   ...     def __str__(self) -> str:
   ...         return self.value
   >>> str(CustomString("hello"))  # 'hello'


.. py:class:: SupportsTypeCheck

   Bases: :py:obj:`Protocol`


   Protocol for types that can validate if a value is of a specific type.

   This protocol defines the interface for objects that can check if a
   value conforms to a particular type specification.


   .. py:method:: is_type(value, target_type)

      Check if a value matches the specified type.

      :param value: The value to check.
      :param target_type: The type to check against.

      :returns: True if the value is of the specified type, False otherwise.
      :rtype: bool



.. py:class:: TypeConverter

   Bases: :py:obj:`Protocol`


   Protocol for types that can convert between different types.

   This protocol defines the interface for objects that implement
   conversion logic between different types.

   .. rubric:: Examples

   >>> class IntToStrConverter:
   ...     def convert(self, value: int) -> str:
   ...         return str(value)
   >>> converter = IntToStrConverter()
   >>> converter.convert(42)  # "42"


   .. py:method:: convert(value)

      Convert a value from one type to another.

      :param value: The value to convert.

      :returns: The converted value.

      :raises TypeError: If the value cannot be converted.
      :raises ValueError: If the value is semantically invalid for conversion.



.. py:class:: TypeDeduplicator

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T`\ ]


   Protocol for objects that can deduplicate values based on type characteristics.

   This protocol defines the interface for objects that can identify and
   remove duplicate values according to type-specific equality criteria.

   Type Args:
       T: The type of values being deduplicated.

   .. rubric:: Examples

   >>> class IntDeduplicator:
   ...     def deduplicate(self, values: list[int]) -> list[int]:
   ...         return list(set(values))
   >>> deduplicator = IntDeduplicator()
   >>> deduplicator.deduplicate([1, 2, 2, 3])  # [1, 2, 3]


   .. py:method:: deduplicate(values)

      Remove duplicate values from a list.

      :param values: The list of values to deduplicate.

      :returns: A new list with duplicates removed.
      :rtype: list[T]



.. py:class:: TypeFactory

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T_co`\ ]


   Protocol for factory objects that create instances of a specific type.

   This protocol defines the interface for objects that can create
   instances of type T_co from various inputs.

   Type Args:
       T_co: The type of object created by the factory (covariant).
             This allows a factory that creates subtypes to be used
             where a factory that creates supertypes is expected.

   .. rubric:: Examples

   >>> class PersonFactory:
   ...     def create(self, name: str, age: int) -> 'Person':
   ...         return Person(name, age)
   >>> factory = PersonFactory()
   >>> person = factory.create("Alice", 30)


   .. py:method:: create(*args, **kwargs)

      Create an instance of type T_co.

      :param \*args: Positional arguments for object construction.
      :param \*\*kwargs: Keyword arguments for object construction.

      :returns: A new instance of type T_co.

      :raises TypeError: If the arguments are incompatible with the type.
      :raises ValueError: If the arguments are semantically invalid.



.. py:class:: TypeForge

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T_co`\ ]


   Protocol for type converters that transform values to specific types.

   Type Args:
       T_co: The target type that values will be converted to (covariant).
             This allows a forge that produces a subtype to be used where
             a forge that produces a supertype is expected.

   .. rubric:: Examples

   >>> class StringForge:
   ...     def forge(self, value: object) -> str:
   ...         return str(value)
   >>> forge = StringForge()
   >>> forge.forge(42)  # "42"


   .. py:method:: forge(value)

      Transforms the given value into the desired type.

      :param value: The value to transform.

      :returns: A value of type T_co.

      :raises TypeError: If the value cannot be converted to type T_co.
      :raises ValueError: If the value is semantically invalid for type T_co.



.. py:class:: TypeInfo

   Bases: :py:obj:`Protocol`


   Protocol for objects that provide type metadata and reflection capabilities.

   This protocol defines the interface for objects that can inspect and
   provide information about types.


   .. py:method:: get_attributes()

      Get the attributes defined by this type.

      :returns: Mapping of attribute names to their types.
      :rtype: dict[str, type]



   .. py:method:: get_name()

      Get the name of the type.

      :returns: The type name.
      :rtype: str



   .. py:method:: is_subtype_of(other)

      Check if this type is a subtype of another type.

      :param other: The potential supertype.

      :returns: True if this type is a subtype of other, False otherwise.
      :rtype: bool



.. py:class:: TypeNormalizer

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T`\ ]


   Protocol for objects that normalize values within a type.

   This protocol defines the interface for objects that transform
   values to a normal form while preserving type.

   Type Args:
       T: The type being normalized.

   .. rubric:: Examples

   >>> class PathNormalizer:
   ...     def normalize(self, value: str) -> str:
   ...         return value.replace('\', '/')
   >>> normalizer = PathNormalizer()
   >>> normalizer.normalize("C:\Windows\System32")  # "C:/Windows/System32"


   .. py:method:: normalize(value)

      Transform a value to its normal form.

      :param value: The value to normalize.

      :returns: The normalized value of the same type.
      :rtype: T

      :raises ValueError: If the value cannot be normalized.



.. py:class:: TypeRegistry

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T`\ ]


   Protocol for type registration and lookup systems.

   This protocol defines the interface for objects that maintain
   a registry of types and associated metadata or factories.

   Type Args:
       T: The type of value associated with each registered type.

   .. rubric:: Examples

   >>> class SimpleTypeRegistry:
   ...     def __init__(self) -> None:
   ...         self._registry: dict[type, str] = {}
   ...     def register(self, cls: type, description: str) -> None:
   ...         self._registry[cls] = description
   ...     def lookup(self, cls: type) -> str:
   ...         return self._registry[cls]
   >>> registry = SimpleTypeRegistry()
   >>> registry.register(int, "Integer type")
   >>> registry.lookup(int)  # "Integer type"


   .. py:method:: lookup(cls)

      Look up the value associated with a type.

      :param cls: The type to look up.

      :returns: The value associated with the type.

      :raises KeyError: If the type is not registered.



   .. py:method:: register(cls, value)

      Register a type with an associated value.

      :param cls: The type to register.
      :param value: The value to associate with the type.



.. py:class:: TypeStandardizer

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.S_contra`\ , :py:obj:`type_forge.typing.variables.T_co`\ ]


   Protocol for objects that standardize values to a canonical form.

   This protocol defines the interface for objects that convert values
   from various forms into a standardized representation.

   Type Args:
       S_contra: The source type (contravariant). This allows a standardizer
                that accepts a supertype to be used where a standardizer that
                accepts a subtype is expected.
       T: The standardized type.

   .. rubric:: Examples

   >>> class CaseStandardizer:
   ...     def standardize(self, value: str) -> str:
   ...         return value.lower()
   >>> standardizer = CaseStandardizer()
   >>> standardizer.standardize("Hello")  # "hello"


   .. py:method:: standardize(value)

      Convert a value to its standardized form.

      :param value: The value to standardize.

      :returns: The standardized value.

      :raises TypeError: If the value cannot be standardized.
      :raises ValueError: If the value is semantically invalid.



.. py:class:: TypedConverter

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.S_contra`\ , :py:obj:`type_forge.typing.variables.T_co`\ ]


   Protocol for types that can convert from a specific type to another specific type.

   This protocol defines the interface for objects that implement
   conversion logic between specific types with strong typing.

   Type Args:
       S_contra: The source type (contravariant). This allows a converter that
                accepts a supertype to be used where a converter that accepts
                a subtype is expected.
       T: The target type.

   .. rubric:: Examples

   >>> class IntToStrConverter:
   ...     def convert(self, value: int) -> str:
   ...         return str(value)
   >>> converter = IntToStrConverter()
   >>> converter.convert(42)  # "42"


   .. py:method:: convert(value)

      Convert a value from type S_contra to type T.

      :param value: The value to convert.

      :returns: The converted value of type T.

      :raises TypeError: If the value cannot be converted.
      :raises ValueError: If the value is semantically invalid for conversion.



.. py:class:: Validator

   Bases: :py:obj:`Protocol`\ [\ :py:obj:`type_forge.typing.variables.T_contra`\ ]


   Protocol for validators that check if values meet certain criteria.

   Validator protocols define the interface for objects that verify
   whether values conform to specific requirements or constraints.

   Type Args:
       T_contra: The type of value being validated (contravariant).
                This allows a validator that can validate a supertype
                to be used where a validator for a subtype is expected.

   .. rubric:: Examples

   >>> class IntValidator:
   ...     def validate(self, value: int) -> bool:
   ...         return value > 0
   >>> validator = IntValidator()
   >>> validator.validate(42)  # True


   .. py:method:: validate(value)

      Validates the given value.

      :param value: The value to validate.

      :returns: True if the value is valid, False otherwise.
      :rtype: bool



.. py:data:: version
   :value: '0.1.0'


