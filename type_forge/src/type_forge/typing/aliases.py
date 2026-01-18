"""
Type Aliases for the Type Forge system.

This module defines a comprehensive type vocabulary that enables precision, safety, and
clarity throughout the Type Forge framework. Each alias has been crafted to provide
maximum type safety, maintaining flexibility through appropriate variance annotations.

Categories:
    - Type identification and naming
    - Registry and mapping structures
    - Field and structure definitions
    - Type relationships and conversion paths
    - Validation and verification mechanisms
    - Collection specializations
    - Error handling and result representations

Each alias embodies Eidosian principles:
that "types are firewalls between intention and mistake."
"""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from .variables import K, K_co, R, S, T, T_co, T_contra, TError, U, U_co, V, V_co

# ──────────────────────────────────────────────────────────────
# Type Aliases with Semantic Meaning
# ──────────────────────────────────────────────────────────────

# Type naming and identification
TypeName = str
"""String identifier for a type (e.g., 'int', 'List[str]').

Used for human-readable type references and registry operations.
"""

TypePath = str
"""Dot-notation path to a type or attribute (e.g., 'module.submodule.MyClass').

Used for dynamic type resolution and attribute access.
"""

ValidationPath = str
"""Path for validation error reporting (e.g., 'user.address.street').

Provides structural context for validation errors in nested data.
"""

ErrorMessage = str
"""Human-readable error message for validation or conversion failures.

Designed for clear communication of errors to end users or developers.
"""

TypeIdentifier = Union[TypeName, Type[object]]
"""Type identified either by name string or class reference.

Allows flexible type specification in APIs that accept either form.
"""

# Registry and mapping types with invariant, covariant and contravariant versions
TypeRegistry = Dict[TypeName, Type[object]]
"""Registry mapping type names to their corresponding type objects.

Foundation for type lookup and registration systems.
"""

TypeRegistryT = Dict[TypeName, Type[T]]
"""Generic type registry constrained to types of T.

Enables type-safe registries for specific base types.
"""

TypeRegistryT_co = Dict[TypeName, Type[T_co]]
"""Covariant type registry allowing subtypes of T_co.

Suitable for registries where subtype polymorphism is desired.
"""

TypeRegistryT_contra = Dict[TypeName, Type[T_contra]]
"""Contravariant type registry accepting supertypes of T_contra.

Specialized registry for cases requiring contravariant behavior.
"""

# Field definitions with invariant, covariant and contravariant versions
FieldDefinitions = Dict[str, Type[object]]
"""Structure field definitions mapping field names to types.

Core definition type for structural type systems.
"""

FieldDefinitionsT = Dict[str, Type[T]]
"""Generic field definitions constrained to types of T.

Enables structure definitions with type constraints.
"""

FieldDefinitionsT_co = Dict[str, Type[T_co]]
"""Covariant field definitions allowing subtypes of T_co.

Supports field definitions with polymorphic type relationships.
"""

FieldDefinitionsT_contra = Dict[str, Type[T_contra]]
"""Contravariant field definitions accepting supertypes of T_contra.

Specialized field definitions for contravariant scenarios.
"""

FieldsWithDefaults = Dict[str, Tuple[Type[object], object]]
"""Fields with default values mapping names to (type, default) pairs.

Supports structural types with default values for optional fields.
"""

FieldsWithDefaultsT = Dict[str, Tuple[Type[T], T]]
"""Generic fields with defaults constrained to specific types.

Type-safe definition of fields with matching default values.
"""

# Type mappings with invariant, covariant and contravariant versions
TypeMap = Dict[Type[object], Type[object]]
"""Mapping between types for conversion or relationship definition.

Foundation for type conversion systems.
"""

TypeMapFrom = Dict[Type[S], Type[object]]
"""Source-specific type map defining conversions from type S.

Specialized mapping for conversions from a specific source type.
"""

TypeMapTo = Dict[Type[object], Type[R]]
"""Target-specific type map defining conversions to type R.

Specialized mapping for conversions to a specific result type.
"""

TypeMapSR = Dict[Type[S], Type[R]]
"""Source-to-Result specific type map with precise typing.

Fully-specified mapping between concrete source and result types.
"""

TypeHierarchy = Dict[Type[object], List[Type[object]]]
"""Type to subtype hierarchy map for inheritance relationships.

Represents inheritance trees for runtime type analysis.
"""

# Converter mappings with precise type specifications
ConverterMap = Dict[Tuple[Type[object], Type[object]], Callable[[object], object]]
"""Type conversion map linking type pairs to conversion functions.

Base converter registry for dynamic type conversion systems.
"""

ConverterMapSR = Dict[Tuple[Type[S], Type[R]], Callable[[S], R]]
"""Source-to-Result converter map with precise typing.

Strongly-typed converter registry for specific type pairs.

Args:
    S: Source type parameter for conversion
    R: Result type parameter for conversion
"""

ConverterMapGeneric = Dict[Tuple[Type[T], Type[U]], Callable[[T], U]]
"""Generic converter map with parametric types.

Flexible converter registry with generic type parameters for type-safe conversion
between any source and destination types. Enables precise typing while maintaining
adaptability for diverse conversion scenarios.

Args:
    T: Source type parameter for conversion input
    U: Target type parameter for conversion output

Example:
    ```python
    converters: ConverterMapGeneric = {
        (int, str): lambda x: str(x),
        (str, int): lambda x: int(x)
    }
    ```
"""

ConverterPriority = Dict[Tuple[Type[object], Type[object]], int]
"""Conversion priority map for resolving ambiguous conversions.

Defines precedence when multiple conversion paths exist.
"""

# Class and validation types
ClassType = Type[T]
"""Generic class type with type parameter T.

Represents classes rather than instances, with type safety.
"""

ClassType_co = Type[T_co]
"""Covariant class type accepting subtypes of T_co.

Supports class hierarchies with covariant relationships.
"""

ClassType_contra = Type[T_contra]
"""Contravariant class type accepting supertypes of T_contra.

Specialized class reference for contravariant scenarios.
"""

# Parent specification types with exact generic parameters
ParentSpecType = Union[ClassType[T], Tuple[ClassType[object], ...]]
"""Parent class specification supporting single class or tuple of classes.

Used for inheritance definitions and interface specifications.
"""

ParentSpecType_co = Union[ClassType[T_co], Tuple[ClassType[object], ...]]
"""Covariant parent specification with subtype support.

Flexibility for polymorphic inheritance relationships.
"""

ParentSpecType_contra = Union[ClassType[T_contra], Tuple[ClassType[object], ...]]
"""Contravariant parent specification for specialized scenarios.

Supports advanced type relationship definitions.
"""

# Validation functions with different signatures
ValidationFunc = Callable[[object], bool]
"""Type validation function for arbitrary objects.

Basic validation with boolean result indicating validity.

Args:
    value (object): Value to validate

Returns:
    bool: True if valid, False otherwise
"""

ValidationFuncT = Callable[[T], bool]
"""Generic validation function for specific types.

Type-safe validation for known input types.

Args:
    value (T): Value of type T to validate

Returns:
    bool: True if valid, False otherwise
"""

ValidationFuncT_contra = Callable[[T_contra], bool]
"""Contravariant validation function accepting supertypes.

Flexible validation function usable with parent types.

Args:
    value (T_contra): Value of type T_contra or any supertype to validate

Returns:
    bool: True if valid, False otherwise
"""

ValidationResultT = Tuple[bool, Optional[ErrorMessage]]
"""Standard validation outcome with validity flag and optional error.

Structured result providing validation status and explanation.

Returns:
    Tuple[bool, Optional[ErrorMessage]]: Validation status and optional error message
"""

ValidationResult = ValidationResultT
"""Primary validation outcome type for consistent usage.

Canonical type for validation results throughout the system.

Returns:
    Tuple[bool, Optional[ErrorMessage]]: Validation status and optional error message
"""

ValidationWithPath = Tuple[bool, Optional[ErrorMessage], Optional[ValidationPath]]
"""Validation with path info for structural validation.

Extended validation result with context path for nested structures.

Returns:
    Tuple[bool, Optional[ErrorMessage], Optional[ValidationPath]]:
        Validation status, optional error message, and optional validation path
"""

# Dictionary and schema types with invariant, covariant and contravariant versions
DictSchemaT = Dict[Hashable, T]
"""Dictionary schema with hashable keys and values of type T.

Foundation for schema-based validation of dictionary-like structures.
"""

DictSchemaT_co = Dict[Hashable, T_co]
"""Covariant dictionary schema allowing subtypes for values.

Flexible schema supporting polymorphic value types.
"""

DictSchemaT_contra = Dict[Hashable, T_contra]
"""Contravariant dictionary schema for specialized scenarios.

Advanced schema type for contravariant validation cases.
"""

# Schema type variants
SchemaTypeT = Union[Dict[Hashable, T], T]
"""Either a schema dictionary or a direct type reference.

Flexible schema specification supporting both simple and complex cases.
"""

SchemaTypeT_co = Union[Dict[Hashable, T_co], T_co]
"""Covariant schema type with subtype support.

Schema specification with polymorphic type relationships.
"""

SchemaTypeT_contra = Union[Dict[Hashable, T_contra], T_contra]
"""Contravariant schema type for specialized scenarios.

Advanced schema specification for contravariant cases.
"""

# Schema value variants
SchemaValueT = Union[Dict[Hashable, object], object]
"""Either a schema dictionary or a direct value.

Represents data that conforms to a schema structure.
"""

SchemaValueT_co = Union[Dict[Hashable, T_co], T_co]
"""Covariant schema value with subtype support.

Flexible value representation for polymorphic scenarios.
"""

SchemaValueT_contra = Union[Dict[Hashable, T_contra], T_contra]
"""Contravariant schema value for specialized scenarios.

Advanced value representation for contravariant cases.
"""

# Collection type groups
CollectionTypes = (list, tuple, set, frozenset, dict)
"""Container types that hold multiple values.

Common Python collection implementations for type checking.
"""

PrimitiveTypes = (int, float, bool, str, bytes)
"""Basic value types directly supported by Python.

Fundamental types with direct language support.
"""

SequenceTypes = (list, tuple)
"""Ordered collection types maintaining element sequence.

Sequential collection types for ordered data.
"""

SetTypes = (set, frozenset)
"""Unordered unique collection types eliminating duplicates.

Set-like collections guaranteeing uniqueness.
"""

MappingTypes = (dict, Mapping)
"""Key-value mapping types for associative data.

Dictionary-like types supporting key-based lookup.
"""

NumericTypes = (int, float, complex)
"""Numeric value types for mathematical operations.

Types supporting arithmetic operations.
"""

TextTypes = (str, bytes)
"""Text value types for character-based data.

Types representing textual information.
"""

IdentityTypes = (int, str)
"""Common identifier types used for entity references.

Types frequently used as unique identifiers.
"""

# Collection-specific type aliases
ListT = List[T]
"""Generic list with elements of type T.

Type-safe list specification for homogeneous elements.
"""

ListT_co = List[T_co]
"""Covariant list allowing subtypes of T_co.

List supporting polymorphic element types.
"""

SetT = Set[T]
"""Generic set with elements of type T.

Type-safe set specification guaranteeing uniqueness.
"""

SetT_co = Set[T_co]
"""Covariant set allowing subtypes of T_co.

Set supporting polymorphic element types.
"""

DictKV = Dict[K, V]
"""Generic dictionary mapping keys of type K to values of type V.

Type-safe dictionary with precise key and value types.
"""

DictKV_co = Dict[K_co, V_co]
"""Covariant dictionary allowing subtype keys and values.

Dictionary supporting polymorphic key and value types.
"""

TupleT = Tuple[T, ...]
"""Generic tuple with elements of type T.

Type-safe immutable sequence with homogeneous elements.
"""

TupleT_co = Tuple[T_co, ...]
"""Covariant tuple allowing subtypes of T_co.

Tuple supporting polymorphic element types.
"""

SequenceT = Sequence[T]
"""Generic sequence with elements of type T.

Abstract sequence type for read-only sequential access.
"""

SequenceT_co = Sequence[T_co]
"""Covariant sequence allowing subtypes of T_co.

Sequence supporting polymorphic element types.
"""

FrozenSetT = FrozenSet[T]
"""Generic frozen set with elements of type T.

Type-safe immutable set with homogeneous elements.
"""

FrozenSetT_co = FrozenSet[T_co]
"""Covariant frozen set allowing subtypes of T_co.

Immutable set supporting polymorphic element types.
"""

IterableT = Iterable[T]
"""Generic iterable with elements of type T.

Abstract iterable type for element traversal.
"""

IterableT_co = Iterable[T_co]
"""Covariant iterable allowing subtypes of T_co.

Iterable supporting polymorphic element types.
"""

IteratorT = Iterator[T]
"""Generic iterator with elements of type T.

Progressive access to elements of a sequence.
"""

IteratorT_co = Iterator[T_co]
"""Covariant iterator allowing subtypes of T_co.

Iterator supporting polymorphic element types.
"""

# Conversion and validation related types with explicit variance
TransformFunc = Callable[[T], U]
"""Transformation function converting type T to type U.

Pure function mapping between types with no side effects.

Args:
    value (T): Input value to transform

Returns:
    U: Transformed output value
"""

TransformFunc_co_contra = Callable[[T_contra], U_co]
"""Contravariant input, covariant output transformation.

Advanced transformation supporting variance at both ends.

Args:
    value (T_contra): Input value (or any supertype) to transform

Returns:
    U_co: Transformed output value (or any subtype)
"""

PredicateFunc = Callable[[T], bool]
"""Boolean predicate function for type T.

Decision function determining if value meets criteria.

Args:
    value (T): Value to test against predicate

Returns:
    bool: True if value meets criteria, False otherwise
"""

PredicateFunc_contra = Callable[[T_contra], bool]
"""Contravariant predicate function accepting supertypes.

Flexible decision function usable with parent types.

Args:
    value (T_contra): Value (or any supertype) to test against predicate

Returns:
    bool: True if value meets criteria, False otherwise
"""

# Type guard and converter functions
TypeGuardFunc = Callable[[object], bool]
"""Type checking function for arbitrary objects.

Runtime type verification for dynamic typing scenarios.

Args:
    value (object): Value to check for type compatibility

Returns:
    bool: True if value matches expected type, False otherwise
"""

TypeGuardFuncT = Callable[[T], bool]
"""Generic type guard function for specific types.

Type-safe verification with known input types.

Args:
    value (T): Value to check for specific type compatibility

Returns:
    bool: True if value matches expected type, False otherwise
"""

TypeConverter = Callable[[S], Optional[R]]
"""Type conversion function transforming S to optional R.

Converts from source type to result type, potentially returning None on failure.

Args:
    value (S): Source value to convert

Returns:
    Optional[R]: Successfully converted value or None if conversion fails
"""

TypeConverterSafe = Callable[[S], R]
"""Non-None type conversion function guaranteeing successful conversion.

Guaranteed conversion from source to result with no failure cases.

Args:
    value (S): Source value to convert

Returns:
    R: Successfully converted value (never None)
"""

OptionalConverter = Callable[[Optional[S]], Optional[R]]
"""Optional-aware converter handling None values appropriately.

Converts optional source values to optional result values.

Args:
    value (Optional[S]): Source value or None

Returns:
    Optional[R]: Converted value or None if input was None/conversion failed
"""

# Error handling and result types
ErrorHandler = Callable[[TError], Optional[T]]
"""Error to result handler recovering from specific errors.

Converts error instances into valid results or None.

Args:
    error (TError): Error instance to handle

Returns:
    Optional[T]: Recovered value or None if recovery impossible
"""

FallbackProvider = Callable[[], T]
"""Provides fallback value when primary operations fail.

Zero-argument function delivering consistent default values.

Returns:
    T: Fallback value of specified type
"""

TryResult = Union[T, TError]
"""Result of operation or error representing success/failure.

Sum type pattern for success/failure outcomes without exceptions.
"""

# Advanced type relationship definitions
TypeRelationship = Literal[
    "identical",
    "subtype",
    "supertype",
    "convertible",
    "incompatible",
]
"""Classification of relationship between two types.

Precise categorization of how types relate to each other:
- identical: Types are exactly the same
- subtype: First type is a subtype of the second
- supertype: First type is a supertype of the second
- convertible: Types can be converted between each other
- incompatible: No relationship exists between types
"""

TypeDistance = int
"""Measure of conversion complexity between types.

Numeric representation of conversion difficulty:
- 0: Identical types (no conversion needed)
- 1: Direct subtype/supertype relationship
- 2+: Increasing complexity of conversion
"""

TypeMatch = Tuple[Type[object], Type[object], TypeRelationship, TypeDistance]
"""Type matching result with relationship classification and distance.

Complete analysis of relationship between two types.
"""

TypePrecedence = Dict[Type[object], int]
"""Type precedence for resolution of ambiguous type situations.

Higher values indicate higher precedence in type selection.
"""

# Advanced validation and verification types
ValidationOptions = Dict[str, object]
"""Options for validation process customization.

Configuration parameters affecting validation behavior.
"""

ValidationStrategy = Callable[[T, ValidationOptions], ValidationResult]
"""Strategic validation with configuration options.

Configurable validation approach for complex scenarios.

Args:
    value (T): Value to validate
    options (ValidationOptions): Configuration parameters

Returns:
    ValidationResult: Validation outcome with status and error
"""

ValidationChain = List[Tuple[ValidationFunc, str]]
"""Chain of validators with associated error messages.

Sequential validation with specific error reporting.
"""

ValidationContext = Dict[str, object]
"""Context for validation process with shared state.

Environmental information affecting validation decisions.
"""
