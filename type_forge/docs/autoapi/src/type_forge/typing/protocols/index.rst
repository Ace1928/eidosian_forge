.. py:module:: src.type_forge.typing.protocols

Type Protocols Module
   ====================

   This module defines protocol classes that specify interfaces for different
   type operations, conversions, and validations in the type forge system.

   These protocols enable static typing and runtime interface verification for
   objects that implement specific behaviors without requiring inheritance.


Module Contents
---------------


   .. py:class:: CompositeValidator   :module: 

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




   .. py:class:: SupportsBoolConversion   :module: 

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




   .. py:class:: SupportsComparison   :module: 

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




   .. py:class:: SupportsEquality   :module: 

      Protocol for objects that support equality operations.




   .. py:class:: SupportsFloat   :module: 

      Protocol for objects that support conversion to float.




   .. py:class:: SupportsFloatConversion   :module: 

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




   .. py:class:: SupportsGetAttr   :module: 

      Protocol for types that support attribute access.

      This protocol defines the interface for objects that implement
      custom attribute access through the __getattr__ method.

      .. rubric:: Examples

      >>> class CustomObject:
      ...     def __getattr__(self, name: str) -> int:
      ...         return len(name)
      >>> obj = CustomObject()
      >>> obj.attribute  # 9




   .. py:class:: SupportsGetItem   :module: 

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




   .. py:class:: SupportsInt   :module: 

      Protocol for objects that support conversion to int.




   .. py:class:: SupportsIntConversion   :module: 

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




   .. py:class:: SupportsIteration   :module: 

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




   .. py:class:: SupportsLen   :module: 

      Protocol for objects that support length operations.




   .. py:class:: SupportsLength   :module: 

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




   .. py:class:: SupportsMapping   :module: 

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




   .. py:class:: SupportsStrConversion   :module: 

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




   .. py:class:: SupportsTypeCheck   :module: 

      Protocol for types that can validate if a value is of a specific type.

      This protocol defines the interface for objects that can check if a
      value conforms to a particular type specification.




   .. py:class:: TypeConverter   :module: 

      Protocol for types that can convert between different types.

      This protocol defines the interface for objects that implement
      conversion logic between different types.

      .. rubric:: Examples

      >>> class IntToStrConverter:
      ...     def convert(self, value: int) -> str:
      ...         return str(value)
      >>> converter = IntToStrConverter()
      >>> converter.convert(42)  # "42"




   .. py:class:: TypeDeduplicator   :module: 

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




   .. py:class:: TypeFactory   :module: 

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




   .. py:class:: TypeForge   :module: 

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




   .. py:class:: TypeInfo   :module: 

      Protocol for objects that provide type metadata and reflection capabilities.

      This protocol defines the interface for objects that can inspect and
      provide information about types.




   .. py:class:: TypeNormalizer   :module: 

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




   .. py:class:: TypeRegistry   :module: 

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




   .. py:class:: TypeStandardizer   :module: 

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




   .. py:class:: TypedConverter   :module: 

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




   .. py:class:: Validator   :module: 

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



.. py:data:: version
      :value: '0.1.0'


