type_forge.typing.aliases
=========================

.. py:module:: type_forge.typing.aliases

.. autoapi-nested-parse::

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



Attributes
----------

.. autoapisummary::

   type_forge.typing.aliases.ClassType
   type_forge.typing.aliases.ClassType_co
   type_forge.typing.aliases.ClassType_contra
   type_forge.typing.aliases.CollectionTypes
   type_forge.typing.aliases.ConverterMap
   type_forge.typing.aliases.ConverterMapGeneric
   type_forge.typing.aliases.ConverterMapSR
   type_forge.typing.aliases.ConverterPriority
   type_forge.typing.aliases.DictKV
   type_forge.typing.aliases.DictKV_co
   type_forge.typing.aliases.DictSchemaT
   type_forge.typing.aliases.DictSchemaT_co
   type_forge.typing.aliases.DictSchemaT_contra
   type_forge.typing.aliases.ErrorHandler
   type_forge.typing.aliases.ErrorMessage
   type_forge.typing.aliases.FallbackProvider
   type_forge.typing.aliases.FieldDefinitions
   type_forge.typing.aliases.FieldDefinitionsT
   type_forge.typing.aliases.FieldDefinitionsT_co
   type_forge.typing.aliases.FieldDefinitionsT_contra
   type_forge.typing.aliases.FieldsWithDefaults
   type_forge.typing.aliases.FieldsWithDefaultsT
   type_forge.typing.aliases.FrozenSetT
   type_forge.typing.aliases.FrozenSetT_co
   type_forge.typing.aliases.IdentityTypes
   type_forge.typing.aliases.IterableT
   type_forge.typing.aliases.IterableT_co
   type_forge.typing.aliases.IteratorT
   type_forge.typing.aliases.IteratorT_co
   type_forge.typing.aliases.ListT
   type_forge.typing.aliases.ListT_co
   type_forge.typing.aliases.MappingTypes
   type_forge.typing.aliases.NumericTypes
   type_forge.typing.aliases.OptionalConverter
   type_forge.typing.aliases.ParentSpecType
   type_forge.typing.aliases.ParentSpecType_co
   type_forge.typing.aliases.ParentSpecType_contra
   type_forge.typing.aliases.PredicateFunc
   type_forge.typing.aliases.PredicateFunc_contra
   type_forge.typing.aliases.PrimitiveTypes
   type_forge.typing.aliases.SchemaTypeT
   type_forge.typing.aliases.SchemaTypeT_co
   type_forge.typing.aliases.SchemaTypeT_contra
   type_forge.typing.aliases.SchemaValueT
   type_forge.typing.aliases.SchemaValueT_co
   type_forge.typing.aliases.SchemaValueT_contra
   type_forge.typing.aliases.SequenceT
   type_forge.typing.aliases.SequenceT_co
   type_forge.typing.aliases.SequenceTypes
   type_forge.typing.aliases.SetT
   type_forge.typing.aliases.SetT_co
   type_forge.typing.aliases.SetTypes
   type_forge.typing.aliases.TextTypes
   type_forge.typing.aliases.TransformFunc
   type_forge.typing.aliases.TransformFunc_co_contra
   type_forge.typing.aliases.TryResult
   type_forge.typing.aliases.TupleT
   type_forge.typing.aliases.TupleT_co
   type_forge.typing.aliases.TypeConverter
   type_forge.typing.aliases.TypeConverterSafe
   type_forge.typing.aliases.TypeDistance
   type_forge.typing.aliases.TypeGuardFunc
   type_forge.typing.aliases.TypeGuardFuncT
   type_forge.typing.aliases.TypeHierarchy
   type_forge.typing.aliases.TypeIdentifier
   type_forge.typing.aliases.TypeMap
   type_forge.typing.aliases.TypeMapFrom
   type_forge.typing.aliases.TypeMapSR
   type_forge.typing.aliases.TypeMapTo
   type_forge.typing.aliases.TypeMatch
   type_forge.typing.aliases.TypeName
   type_forge.typing.aliases.TypePath
   type_forge.typing.aliases.TypePrecedence
   type_forge.typing.aliases.TypeRegistry
   type_forge.typing.aliases.TypeRegistryT
   type_forge.typing.aliases.TypeRegistryT_co
   type_forge.typing.aliases.TypeRegistryT_contra
   type_forge.typing.aliases.TypeRelationship
   type_forge.typing.aliases.ValidationChain
   type_forge.typing.aliases.ValidationContext
   type_forge.typing.aliases.ValidationFunc
   type_forge.typing.aliases.ValidationFuncT
   type_forge.typing.aliases.ValidationFuncT_contra
   type_forge.typing.aliases.ValidationOptions
   type_forge.typing.aliases.ValidationPath
   type_forge.typing.aliases.ValidationResult
   type_forge.typing.aliases.ValidationResultT
   type_forge.typing.aliases.ValidationStrategy
   type_forge.typing.aliases.ValidationWithPath


Module Contents
---------------

.. py:data:: ClassType

   Generic class type with type parameter T.

   Represents classes rather than instances, with type safety.

.. py:data:: ClassType_co

   Covariant class type accepting subtypes of T_co.

   Supports class hierarchies with covariant relationships.

.. py:data:: ClassType_contra

   Contravariant class type accepting supertypes of T_contra.

   Specialized class reference for contravariant scenarios.

.. py:data:: CollectionTypes

   Container types that hold multiple values.

   Common Python collection implementations for type checking.

.. py:data:: ConverterMap

   Type conversion map linking type pairs to conversion functions.

   Base converter registry for dynamic type conversion systems.

.. py:data:: ConverterMapGeneric

   Generic converter map with parametric types.

   Flexible converter registry with generic type parameters for type-safe conversion
   between any source and destination types. Enables precise typing while maintaining
   adaptability for diverse conversion scenarios.

   :param T: Source type parameter for conversion input
   :param U: Target type parameter for conversion output

   .. admonition:: Example

      ```python
      converters: ConverterMapGeneric = {
          (int, str): lambda x: str(x),
          (str, int): lambda x: int(x)
      }
      ```

.. py:data:: ConverterMapSR

   Source-to-Result converter map with precise typing.

   Strongly-typed converter registry for specific type pairs.

   :param S: Source type parameter for conversion
   :param R: Result type parameter for conversion

.. py:data:: ConverterPriority

   Conversion priority map for resolving ambiguous conversions.

   Defines precedence when multiple conversion paths exist.

.. py:data:: DictKV

   Generic dictionary mapping keys of type K to values of type V.

   Type-safe dictionary with precise key and value types.

.. py:data:: DictKV_co

   Covariant dictionary allowing subtype keys and values.

   Dictionary supporting polymorphic key and value types.

.. py:data:: DictSchemaT

   Dictionary schema with hashable keys and values of type T.

   Foundation for schema-based validation of dictionary-like structures.

.. py:data:: DictSchemaT_co

   Covariant dictionary schema allowing subtypes for values.

   Flexible schema supporting polymorphic value types.

.. py:data:: DictSchemaT_contra

   Contravariant dictionary schema for specialized scenarios.

   Advanced schema type for contravariant validation cases.

.. py:data:: ErrorHandler

   Error to result handler recovering from specific errors.

   Converts error instances into valid results or None.

   :param error: Error instance to handle
   :type error: TError

   :returns: Recovered value or None if recovery impossible
   :rtype: Optional[T]

.. py:data:: ErrorMessage

   Human-readable error message for validation or conversion failures.

   Designed for clear communication of errors to end users or developers.

.. py:data:: FallbackProvider

   Provides fallback value when primary operations fail.

   Zero-argument function delivering consistent default values.

   :returns: Fallback value of specified type
   :rtype: T

.. py:data:: FieldDefinitions

   Structure field definitions mapping field names to types.

   Core definition type for structural type systems.

.. py:data:: FieldDefinitionsT

   Generic field definitions constrained to types of T.

   Enables structure definitions with type constraints.

.. py:data:: FieldDefinitionsT_co

   Covariant field definitions allowing subtypes of T_co.

   Supports field definitions with polymorphic type relationships.

.. py:data:: FieldDefinitionsT_contra

   Contravariant field definitions accepting supertypes of T_contra.

   Specialized field definitions for contravariant scenarios.

.. py:data:: FieldsWithDefaults

   Fields with default values mapping names to (type, default) pairs.

   Supports structural types with default values for optional fields.

.. py:data:: FieldsWithDefaultsT

   Generic fields with defaults constrained to specific types.

   Type-safe definition of fields with matching default values.

.. py:data:: FrozenSetT

   Generic frozen set with elements of type T.

   Type-safe immutable set with homogeneous elements.

.. py:data:: FrozenSetT_co

   Covariant frozen set allowing subtypes of T_co.

   Immutable set supporting polymorphic element types.

.. py:data:: IdentityTypes

   Common identifier types used for entity references.

   Types frequently used as unique identifiers.

.. py:data:: IterableT

   Generic iterable with elements of type T.

   Abstract iterable type for element traversal.

.. py:data:: IterableT_co

   Covariant iterable allowing subtypes of T_co.

   Iterable supporting polymorphic element types.

.. py:data:: IteratorT

   Generic iterator with elements of type T.

   Progressive access to elements of a sequence.

.. py:data:: IteratorT_co

   Covariant iterator allowing subtypes of T_co.

   Iterator supporting polymorphic element types.

.. py:data:: ListT

   Generic list with elements of type T.

   Type-safe list specification for homogeneous elements.

.. py:data:: ListT_co

   Covariant list allowing subtypes of T_co.

   List supporting polymorphic element types.

.. py:data:: MappingTypes

   Key-value mapping types for associative data.

   Dictionary-like types supporting key-based lookup.

.. py:data:: NumericTypes

   Numeric value types for mathematical operations.

   Types supporting arithmetic operations.

.. py:data:: OptionalConverter

   Optional-aware converter handling None values appropriately.

   Converts optional source values to optional result values.

   :param value: Source value or None
   :type value: Optional[S]

   :returns: Converted value or None if input was None/conversion failed
   :rtype: Optional[R]

.. py:data:: ParentSpecType

   Parent class specification supporting single class or tuple of classes.

   Used for inheritance definitions and interface specifications.

.. py:data:: ParentSpecType_co

   Covariant parent specification with subtype support.

   Flexibility for polymorphic inheritance relationships.

.. py:data:: ParentSpecType_contra

   Contravariant parent specification for specialized scenarios.

   Supports advanced type relationship definitions.

.. py:data:: PredicateFunc

   Boolean predicate function for type T.

   Decision function determining if value meets criteria.

   :param value: Value to test against predicate
   :type value: T

   :returns: True if value meets criteria, False otherwise
   :rtype: bool

.. py:data:: PredicateFunc_contra

   Contravariant predicate function accepting supertypes.

   Flexible decision function usable with parent types.

   :param value: Value (or any supertype) to test against predicate
   :type value: T_contra

   :returns: True if value meets criteria, False otherwise
   :rtype: bool

.. py:data:: PrimitiveTypes

   Basic value types directly supported by Python.

   Fundamental types with direct language support.

.. py:data:: SchemaTypeT

   Either a schema dictionary or a direct type reference.

   Flexible schema specification supporting both simple and complex cases.

.. py:data:: SchemaTypeT_co

   Covariant schema type with subtype support.

   Schema specification with polymorphic type relationships.

.. py:data:: SchemaTypeT_contra

   Contravariant schema type for specialized scenarios.

   Advanced schema specification for contravariant cases.

.. py:data:: SchemaValueT

   Either a schema dictionary or a direct value.

   Represents data that conforms to a schema structure.

.. py:data:: SchemaValueT_co

   Covariant schema value with subtype support.

   Flexible value representation for polymorphic scenarios.

.. py:data:: SchemaValueT_contra

   Contravariant schema value for specialized scenarios.

   Advanced value representation for contravariant cases.

.. py:data:: SequenceT

   Generic sequence with elements of type T.

   Abstract sequence type for read-only sequential access.

.. py:data:: SequenceT_co

   Covariant sequence allowing subtypes of T_co.

   Sequence supporting polymorphic element types.

.. py:data:: SequenceTypes

   Ordered collection types maintaining element sequence.

   Sequential collection types for ordered data.

.. py:data:: SetT

   Generic set with elements of type T.

   Type-safe set specification guaranteeing uniqueness.

.. py:data:: SetT_co

   Covariant set allowing subtypes of T_co.

   Set supporting polymorphic element types.

.. py:data:: SetTypes

   Unordered unique collection types eliminating duplicates.

   Set-like collections guaranteeing uniqueness.

.. py:data:: TextTypes

   Text value types for character-based data.

   Types representing textual information.

.. py:data:: TransformFunc

   Transformation function converting type T to type U.

   Pure function mapping between types with no side effects.

   :param value: Input value to transform
   :type value: T

   :returns: Transformed output value
   :rtype: U

.. py:data:: TransformFunc_co_contra

   Contravariant input, covariant output transformation.

   Advanced transformation supporting variance at both ends.

   :param value: Input value (or any supertype) to transform
   :type value: T_contra

   :returns: Transformed output value (or any subtype)
   :rtype: U_co

.. py:data:: TryResult

   Result of operation or error representing success/failure.

   Sum type pattern for success/failure outcomes without exceptions.

.. py:data:: TupleT

   Generic tuple with elements of type T.

   Type-safe immutable sequence with homogeneous elements.

.. py:data:: TupleT_co

   Covariant tuple allowing subtypes of T_co.

   Tuple supporting polymorphic element types.

.. py:data:: TypeConverter

   Type conversion function transforming S to optional R.

   Converts from source type to result type, potentially returning None on failure.

   :param value: Source value to convert
   :type value: S

   :returns: Successfully converted value or None if conversion fails
   :rtype: Optional[R]

.. py:data:: TypeConverterSafe

   Non-None type conversion function guaranteeing successful conversion.

   Guaranteed conversion from source to result with no failure cases.

   :param value: Source value to convert
   :type value: S

   :returns: Successfully converted value (never None)
   :rtype: R

.. py:data:: TypeDistance

   Measure of conversion complexity between types.

   Numeric representation of conversion difficulty:
   - 0: Identical types (no conversion needed)
   - 1: Direct subtype/supertype relationship
   - 2+: Increasing complexity of conversion

.. py:data:: TypeGuardFunc

   Type checking function for arbitrary objects.

   Runtime type verification for dynamic typing scenarios.

   :param value: Value to check for type compatibility
   :type value: object

   :returns: True if value matches expected type, False otherwise
   :rtype: bool

.. py:data:: TypeGuardFuncT

   Generic type guard function for specific types.

   Type-safe verification with known input types.

   :param value: Value to check for specific type compatibility
   :type value: T

   :returns: True if value matches expected type, False otherwise
   :rtype: bool

.. py:data:: TypeHierarchy

   Type to subtype hierarchy map for inheritance relationships.

   Represents inheritance trees for runtime type analysis.

.. py:data:: TypeIdentifier

   Type identified either by name string or class reference.

   Allows flexible type specification in APIs that accept either form.

.. py:data:: TypeMap

   Mapping between types for conversion or relationship definition.

   Foundation for type conversion systems.

.. py:data:: TypeMapFrom

   Source-specific type map defining conversions from type S.

   Specialized mapping for conversions from a specific source type.

.. py:data:: TypeMapSR

   Source-to-Result specific type map with precise typing.

   Fully-specified mapping between concrete source and result types.

.. py:data:: TypeMapTo

   Target-specific type map defining conversions to type R.

   Specialized mapping for conversions to a specific result type.

.. py:data:: TypeMatch

   Type matching result with relationship classification and distance.

   Complete analysis of relationship between two types.

.. py:data:: TypeName

   String identifier for a type (e.g., 'int', 'List[str]').

   Used for human-readable type references and registry operations.

.. py:data:: TypePath

   Dot-notation path to a type or attribute (e.g., 'module.submodule.MyClass').

   Used for dynamic type resolution and attribute access.

.. py:data:: TypePrecedence

   Type precedence for resolution of ambiguous type situations.

   Higher values indicate higher precedence in type selection.

.. py:data:: TypeRegistry

   Registry mapping type names to their corresponding type objects.

   Foundation for type lookup and registration systems.

.. py:data:: TypeRegistryT

   Generic type registry constrained to types of T.

   Enables type-safe registries for specific base types.

.. py:data:: TypeRegistryT_co

   Covariant type registry allowing subtypes of T_co.

   Suitable for registries where subtype polymorphism is desired.

.. py:data:: TypeRegistryT_contra

   Contravariant type registry accepting supertypes of T_contra.

   Specialized registry for cases requiring contravariant behavior.

.. py:data:: TypeRelationship

   Classification of relationship between two types.

   Precise categorization of how types relate to each other:
   - identical: Types are exactly the same
   - subtype: First type is a subtype of the second
   - supertype: First type is a supertype of the second
   - convertible: Types can be converted between each other
   - incompatible: No relationship exists between types

.. py:data:: ValidationChain

   Chain of validators with associated error messages.

   Sequential validation with specific error reporting.

.. py:data:: ValidationContext

   Context for validation process with shared state.

   Environmental information affecting validation decisions.

.. py:data:: ValidationFunc

   Type validation function for arbitrary objects.

   Basic validation with boolean result indicating validity.

   :param value: Value to validate
   :type value: object

   :returns: True if valid, False otherwise
   :rtype: bool

.. py:data:: ValidationFuncT

   Generic validation function for specific types.

   Type-safe validation for known input types.

   :param value: Value of type T to validate
   :type value: T

   :returns: True if valid, False otherwise
   :rtype: bool

.. py:data:: ValidationFuncT_contra

   Contravariant validation function accepting supertypes.

   Flexible validation function usable with parent types.

   :param value: Value of type T_contra or any supertype to validate
   :type value: T_contra

   :returns: True if valid, False otherwise
   :rtype: bool

.. py:data:: ValidationOptions

   Options for validation process customization.

   Configuration parameters affecting validation behavior.

.. py:data:: ValidationPath

   Path for validation error reporting (e.g., 'user.address.street').

   Provides structural context for validation errors in nested data.

.. py:data:: ValidationResult

   Primary validation outcome type for consistent usage.

   Canonical type for validation results throughout the system.

   :returns: Validation status and optional error message
   :rtype: Tuple[bool, Optional[ErrorMessage]]

.. py:data:: ValidationResultT

   Standard validation outcome with validity flag and optional error.

   Structured result providing validation status and explanation.

   :returns: Validation status and optional error message
   :rtype: Tuple[bool, Optional[ErrorMessage]]

.. py:data:: ValidationStrategy

   Strategic validation with configuration options.

   Configurable validation approach for complex scenarios.

   :param value: Value to validate
   :type value: T
   :param options: Configuration parameters
   :type options: ValidationOptions

   :returns: Validation outcome with status and error
   :rtype: ValidationResult

.. py:data:: ValidationWithPath

   Validation with path info for structural validation.

   Extended validation result with context path for nested structures.

   :returns:     Validation status, optional error message, and optional validation path
   :rtype: Tuple[bool, Optional[ErrorMessage], Optional[ValidationPath]]

