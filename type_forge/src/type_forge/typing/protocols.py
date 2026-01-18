# filepath: /home/lloyd/eidosian_forge/type_forge/src/type_forge/typing/protocols.py
"""
Type Protocols Module
====================

This module defines protocol classes that specify interfaces for different
type operations, conversions, and validations in the type forge system.

These protocols enable static typing and runtime interface verification for
objects that implement specific behaviors without requiring inheritance.
"""

from typing import Hashable, Iterator, Protocol, runtime_checkable

from type_forge.typing.variables import (
    K_co,
    K_contra,
    S_contra,
    T,
    T_co,
    T_contra,
    V,
    V_co,
)

from . import __version__  # noqa: F401

version = __version__


# ──────────────────────────────────────────────────────────────
# Type Protocol Definitions
# ──────────────────────────────────────────────────────────────
# Protocol definitions


@runtime_checkable
class SupportsLen(Protocol):
    """Protocol for objects that support length operations."""

    def __len__(self) -> int: ...


@runtime_checkable
class SupportsEquality(Protocol):
    """Protocol for objects that support equality operations."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


@runtime_checkable
class SupportsInt(Protocol):
    """Protocol for objects that support conversion to int."""

    def __int__(self) -> int: ...


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for objects that support conversion to float."""

    def __float__(self) -> float: ...


@runtime_checkable
class SupportsIntConversion(Protocol):
    """Protocol for types that can be converted to int.

    This protocol defines the interface for objects that support
    conversion to integer values through the __int__ method.

    Examples:
        >>> class CustomInteger:
        ...     def __init__(self, value: int) -> None:
        ...         self.value = value
        ...     def __int__(self) -> int:
        ...         return self.value
        >>> isinstance(CustomInteger(42), SupportsIntConversion)  # True at runtime
    """

    def __int__(self) -> int:
        """Convert to integer.

        Returns:
            int: Integer representation of the object.
        """
        ...


@runtime_checkable
class SupportsBoolConversion(Protocol):
    """Protocol for types that can be converted to bool.

    This protocol defines the interface for objects that support
    conversion to boolean values through the __bool__ method.

    Examples:
        >>> class CustomBoolean:
        ...     def __init__(self, value: bool) -> None:
        ...         self.value = value
        ...     def __bool__(self) -> bool:
        ...         return self.value
        >>> bool(CustomBoolean(True))  # True
    """

    def __bool__(self) -> bool:
        """Convert to boolean.

        Returns:
            bool: Boolean representation of the object.
        """
        ...


@runtime_checkable
class SupportsStrConversion(Protocol):
    """Protocol for types that can be converted to str.

    This protocol defines the interface for objects that support
    conversion to string values through the __str__ method.

    Examples:
        >>> class CustomString:
        ...     def __init__(self, value: str) -> None:
        ...         self.value = value
        ...     def __str__(self) -> str:
        ...         return self.value
        >>> str(CustomString("hello"))  # 'hello'
    """

    def __str__(self) -> str:
        """Convert to string.

        Returns:
            str: String representation of the object.
        """
        ...


@runtime_checkable
class SupportsFloatConversion(Protocol):
    """Protocol for types that can be converted to float.

    This protocol defines the interface for objects that support
    conversion to floating-point values through the __float__ method.

    Examples:
        >>> class CustomFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self.value = value
        ...     def __float__(self) -> float:
        ...         return self.value
        >>> float(CustomFloat(3.14))  # 3.14
    """

    def __float__(self) -> float:
        """Convert to float.

        Returns:
            float: Floating-point representation of the object.
        """
        ...


@runtime_checkable
class SupportsComparison(Protocol):
    """Protocol for types that support comparison operations.

    This protocol defines the interface for objects that can be compared
    using standard comparison operators.

    Examples:
        >>> class ComparableValue:
        ...     def __init__(self, value: int) -> None:
        ...         self.value = value
        ...     def __lt__(self, other: object) -> bool:
        ...         if isinstance(other, ComparableValue):
        ...             return self.value < other.value
        ...         return NotImplemented
        >>> ComparableValue(1) < ComparableValue(2)  # True
    """

    def __lt__(self, other: object) -> bool:
        """Compare if self is less than other.

        Args:
            other: Object to compare against.

        Returns:
            bool: True if self is less than other, False otherwise.
        """
        ...


@runtime_checkable
class SupportsIteration(Protocol[T_co]):
    """Protocol for types that support iteration.

    This protocol defines the interface for objects that can be iterated over
    using a for loop or with the iter() function.

    Type Args:
        T_co: The type of items yielded by the iterator (covariant).
              This allows an iterator of a subtype to be used where an
              iterator of a supertype is expected.

    Examples:
        >>> class CustomIterable:
        ...     def __init__(self, values: list[int]) -> None:
        ...         self.values = values
        ...     def __iter__(self) -> Iterator[int]:
        ...         return iter(self.values)
        >>> list(CustomIterable([1, 2, 3]))  # [1, 2, 3]
    """

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator for this object.

        Returns:
            Iterator[T_co]: An iterator yielding items of type T_co.
        """
        ...


@runtime_checkable
class SupportsLength(Protocol):
    """Protocol for types that support the len() function.

    This protocol defines the interface for objects that can report
    their length through the __len__ method.

    Examples:
        >>> class CustomSized:
        ...     def __init__(self, size: int) -> None:
        ...         self.size = size
        ...     def __len__(self) -> int:
        ...         return self.size
        >>> len(CustomSized(5))  # 5
    """

    def __len__(self) -> int:
        """Return the length of the object.

        Returns:
            int: The number of items in the object.
        """
        ...


@runtime_checkable
class SupportsGetItem(Protocol[K_contra, V_co]):
    """Protocol for types that support item access with square brackets.

    This protocol defines the interface for objects that can be accessed
    using the subscription notation (obj[key]).

    Type Args:
        K_contra: The key type (contravariant). This allows a container that
                 accepts a supertype to be used where a container that accepts
                 a subtype is expected.
        V_co: The value type (covariant). This allows a container that returns
              a subtype to be used where a container that returns a supertype
              is expected.

    Examples:
        >>> class CustomContainer:
        ...     def __init__(self, data: dict[str, int]) -> None:
        ...         self.data = data
        ...     def __getitem__(self, key: str) -> int:
        ...         return self.data[key]
        >>> container = CustomContainer({"one": 1})
        >>> container["one"]  # 1
    """

    def __getitem__(self, key: K_contra) -> V_co:
        """Get item at the specified key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key is not found.
        """
        ...


@runtime_checkable
class SupportsGetAttr(Protocol):
    """Protocol for types that support attribute access.

    This protocol defines the interface for objects that implement
    custom attribute access through the __getattr__ method.

    Examples:
        >>> class CustomObject:
        ...     def __getattr__(self, name: str) -> int:
        ...         return len(name)
        >>> obj = CustomObject()
        >>> obj.attribute  # 9
    """

    def __getattr__(self, name: str) -> object:
        """Get attribute by name.

        Args:
            name: Name of the attribute to retrieve.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute is not found.
        """
        ...


@runtime_checkable
class Validator(Protocol[T_contra]):
    """Protocol for validators that check if values meet certain criteria.

    Validator protocols define the interface for objects that verify
    whether values conform to specific requirements or constraints.

    Type Args:
        T_contra: The type of value being validated (contravariant).
                 This allows a validator that can validate a supertype
                 to be used where a validator for a subtype is expected.

    Examples:
        >>> class IntValidator:
        ...     def validate(self, value: int) -> bool:
        ...         return value > 0
        >>> validator = IntValidator()
        >>> validator.validate(42)  # True
    """

    def validate(self, value: T_contra) -> bool:
        """Validates the given value.

        Args:
            value: The value to validate.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        ...


@runtime_checkable
class TypeForge(Protocol[T_co]):
    """Protocol for type converters that transform values to specific types.

    Type Args:
        T_co: The target type that values will be converted to (covariant).
              This allows a forge that produces a subtype to be used where
              a forge that produces a supertype is expected.

    Examples:
        >>> class StringForge:
        ...     def forge(self, value: object) -> str:
        ...         return str(value)
        >>> forge = StringForge()
        >>> forge.forge(42)  # "42"
    """

    def forge(self, value: object) -> T_co:
        """Transforms the given value into the desired type.

        Args:
            value: The value to transform.

        Returns:
            A value of type T_co.

        Raises:
            TypeError: If the value cannot be converted to type T_co.
            ValueError: If the value is semantically invalid for type T_co.
        """
        ...


@runtime_checkable
class TypeConverter(Protocol):
    """Protocol for types that can convert between different types.

    This protocol defines the interface for objects that implement
    conversion logic between different types.

    Examples:
        >>> class IntToStrConverter:
        ...     def convert(self, value: int) -> str:
        ...         return str(value)
        >>> converter = IntToStrConverter()
        >>> converter.convert(42)  # "42"
    """

    def convert(self, value: object) -> object:
        """Convert a value from one type to another.

        Args:
            value: The value to convert.

        Returns:
            The converted value.

        Raises:
            TypeError: If the value cannot be converted.
            ValueError: If the value is semantically invalid for conversion.
        """
        ...


@runtime_checkable
class TypeFactory(Protocol[T_co]):
    """Protocol for factory objects that create instances of a specific type.

    This protocol defines the interface for objects that can create
    instances of type T_co from various inputs.

    Type Args:
        T_co: The type of object created by the factory (covariant).
              This allows a factory that creates subtypes to be used
              where a factory that creates supertypes is expected.

    Examples:
        >>> class PersonFactory:
        ...     def create(self, name: str, age: int) -> 'Person':
        ...         return Person(name, age)
        >>> factory = PersonFactory()
        >>> person = factory.create("Alice", 30)
    """

    def create(self, *args: object, **kwargs: object) -> T_co:
        """Create an instance of type T_co.

        Args:
            *args: Positional arguments for object construction.
            **kwargs: Keyword arguments for object construction.

        Returns:
            A new instance of type T_co.

        Raises:
            TypeError: If the arguments are incompatible with the type.
            ValueError: If the arguments are semantically invalid.
        """
        ...


@runtime_checkable
class SupportsTypeCheck(Protocol):
    """Protocol for types that can validate if a value is of a specific type.

    This protocol defines the interface for objects that can check if a
    value conforms to a particular type specification.
    """

    def is_type(self, value: object, target_type: type) -> bool:
        """Check if a value matches the specified type.

        Args:
            value: The value to check.
            target_type: The type to check against.

        Returns:
            bool: True if the value is of the specified type, False otherwise.
        """
        ...


@runtime_checkable
class TypeInfo(Protocol):
    """Protocol for objects that provide type metadata and reflection capabilities.

    This protocol defines the interface for objects that can inspect and
    provide information about types.
    """

    def get_name(self) -> str:
        """Get the name of the type.

        Returns:
            str: The type name.
        """
        ...

    def is_subtype_of(self, other: type) -> bool:
        """Check if this type is a subtype of another type.

        Args:
            other: The potential supertype.

        Returns:
            bool: True if this type is a subtype of other, False otherwise.
        """
        ...

    def get_attributes(self) -> dict[str, type]:
        """Get the attributes defined by this type.

        Returns:
            dict[str, type]: Mapping of attribute names to their types.
        """
        ...


@runtime_checkable
class TypeNormalizer(Protocol[T]):
    """Protocol for objects that normalize values within a type.

    This protocol defines the interface for objects that transform
    values to a normal form while preserving type.

    Type Args:
        T: The type being normalized.

    Examples:
        >>> class PathNormalizer:
        ...     def normalize(self, value: str) -> str:
        ...         return value.replace('\\', '/')
        >>> normalizer = PathNormalizer()
        >>> normalizer.normalize("C:\\Windows\\System32")  # "C:/Windows/System32"
    """

    def normalize(self, value: T) -> T:
        """Transform a value to its normal form.

        Args:
            value: The value to normalize.

        Returns:
            T: The normalized value of the same type.

        Raises:
            ValueError: If the value cannot be normalized.
        """
        ...


@runtime_checkable
class CompositeValidator(Protocol[T]):
    """Protocol for validators that combine multiple validation rules.

    Type Args:
        T: The type of value being validated.
            For proper variance handling, use this protocol with specific types.

    Examples:
        >>> class AndValidator:
        ...     def __init__(self) -> None:
        ...         self.validators = []
        ...     def add_validator(self, validator):
        ...         self.validators.append(validator)
        ...     def validate(self, value):
        ...         return all(v.validate(value) for v in self.validators)
    """

    def add_validator(self, validator: Validator[T]) -> None:
        """Adds a validator to the composite validator.

        Args:
            validator: The validator to add.
        """
        ...

    def validate(self, value: T) -> bool:
        """Validates the given value using all added validators.

        Args:
            value: The value to validate.

        Returns:
            bool: True if the value passes all validators, False otherwise.
        """
        ...


@runtime_checkable
class SupportsMapping(Protocol[K_co, V]):
    """Protocol for types that support dictionary-like mapping operations.

    This protocol defines the interface for objects that implement mapping
    behavior with key-value pairs.

    Type Args:
        K: The key type.
        V: The value type.

    Examples:
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
    """

    def __getitem__(self, key: Hashable) -> V:
        """Get the value for a given key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key is not found.
        """
        ...

    def __setitem__(self, key: Hashable, value: V) -> None:
        """Set the value for a given key.

        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        ...

    def __contains__(self, key: object) -> bool:
        """Check if the mapping contains the specified key.

        Args:
            key: The key to check for.

        Returns:
            bool: True if the key exists in the mapping, False otherwise.
        """
        ...


@runtime_checkable
class TypedConverter(Protocol[S_contra, T_co]):
    """Protocol for types that can convert from a specific type to another type.

    This protocol defines the interface for objects that implement
    conversion logic between specific types with strong typing.

    Type Args:
        S_contra: The source type (contravariant). This allows a converter that
                 accepts a supertype to be used where a converter that accepts
                 a subtype is expected.
        T: The target type.

    Examples:
        >>> class IntToStrConverter:
        ...     def convert(self, value: int) -> str:
        ...         return str(value)
        >>> converter = IntToStrConverter()
        >>> converter.convert(42)  # "42"
    """

    def convert(self, value: S_contra) -> object:
        """Convert a value from type S_contra to type T.

        Args:
            value: The value to convert.

        Returns:
            The converted value of type T.

        Raises:
            TypeError: If the value cannot be converted.
            ValueError: If the value is semantically invalid for conversion.
        """
        ...


@runtime_checkable
class TypeRegistry(Protocol[T]):
    """Protocol for type registration and lookup systems.

    This protocol defines the interface for objects that maintain
    a registry of types and associated metadata or factories.

    Type Args:
        T: The type of value associated with each registered type.

    Examples:
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
    """

    def register(self, cls: type, value: T) -> None:
        """Register a type with an associated value.

        Args:
            cls: The type to register.
            value: The value to associate with the type.
        """
        ...

    def lookup(self, cls: type) -> T:
        """Look up the value associated with a type.

        Args:
            cls: The type to look up.

        Returns:
            The value associated with the type.

        Raises:
            KeyError: If the type is not registered.
        """
        ...


@runtime_checkable
class TypeDeduplicator(Protocol[T]):
    """Protocol for objects that can deduplicate values based on type characteristics.

    This protocol defines the interface for objects that can identify and
    remove duplicate values according to type-specific equality criteria.

    Type Args:
        T: The type of values being deduplicated.

    Examples:
        >>> class IntDeduplicator:
        ...     def deduplicate(self, values: list[int]) -> list[int]:
        ...         return list(set(values))
        >>> deduplicator = IntDeduplicator()
        >>> deduplicator.deduplicate([1, 2, 2, 3])  # [1, 2, 3]
    """

    def deduplicate(self, values: list[T]) -> list[T]:
        """Remove duplicate values from a list.

        Args:
            values: The list of values to deduplicate.

        Returns:
            list[T]: A new list with duplicates removed.
        """
        ...


@runtime_checkable
class TypeStandardizer(Protocol[S_contra, T_co]):
    """Protocol for objects that standardize values to a canonical form.

    This protocol defines the interface for objects that convert values
    from various forms into a standardized representation.

    Type Args:
        S_contra: The source type (contravariant). This allows a standardizer
                 that accepts a supertype to be used where a standardizer that
                 accepts a subtype is expected.
        T: The standardized type.

    Examples:
        >>> class CaseStandardizer:
        ...     def standardize(self, value: str) -> str:
        ...         return value.lower()
        >>> standardizer = CaseStandardizer()
        >>> standardizer.standardize("Hello")  # "hello"
    """

    def standardize(self, value: S_contra) -> object:
        """Convert a value to its standardized form.

        Args:
            value: The value to standardize.

        Returns:
            The standardized value.

        Raises:
            TypeError: If the value cannot be standardized.
            ValueError: If the value is semantically invalid.
        """
        ...
