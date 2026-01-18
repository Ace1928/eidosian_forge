"""Type Forge Typing System
==========================

This module provides a comprehensive set of typing utilities for precise type
manipulation, validation, conversion, and analysis within the Type Forge ecosystem.

Core Components
--------------

The typing system is organized into several interconnected modules that form a complete
type management ecosystem:

- `aliases`: Type aliases with semantic meaning for enhanced code readability
- `analysis`: Type relationship analysis and compatibility determination
- `conversion`: Type transformation and coercion functions with elegant error handling
- `definitions`: Core type structures, enumerations, and semantic categories
- `hints`: Advanced type hints for complex nested structures and schema validation
- `mapping`: Type relationship analysis, classification, and taxonomy functions
- `naming`: Standardized type name generation and representation utilities
- `protocols`: Protocol interfaces defining type behaviors and operational contracts
- `standardization`: Type normalization, deduplication, and standardization
- `validation`: Type validation and verification utilities with static guarantees
- `variables`: Generic type variables with precise variance annotations

Features
--------

- **Type Safety**: Complete static typing with no `Any` types
- **Elegant Error Handling**: Monadic-style error handling for conversions
- **Precise Relationship Mapping**: Comprehensive type relationship analysis
- **Robust Validation**: Multiple validation strategies with configurable severity
- **Protocol-Based Behavior**: Interface definitions for type-driven operations
- **Self-Documented Structure**: Clear categorical organization with recursive precision

These components enable recursively self-improving type operations that maintain
integrity across all abstraction layers.

Examples
--------
>>> from type_forge.typing import try_convert, describe_type
>>> result = try_convert("42", int)
>>> assert result.success and result.value == 42
>>> type_info = describe_type(list[str])
>>> assert type_info.category == TypeCategory.COLLECTION

Notes
-----
All functions in this module are designed with strict typing and comprehensive
error handling to ensure maximum reliability in type-sensitive operations.
"""

from __future__ import annotations

import sys

# ===============================================================================
# Type Aliases - Semantic type definitions for enhanced readability
# ===============================================================================
from type_forge.typing.aliases import (
    CollectionTypes,
    ConverterMap,
    ConverterMapGeneric,
    ConverterMapSR,
    ConverterPriority,
    DictKV,
    DictKV_co,
    DictSchemaT,
    DictSchemaT_co,
    DictSchemaT_contra,
    ErrorHandler,
    ErrorMessage,
    FallbackProvider,
    FieldDefinitions,
    FieldDefinitionsT,
    FieldDefinitionsT_co,
    FieldDefinitionsT_contra,
    FieldsWithDefaults,
    FieldsWithDefaultsT,
    FrozenSetT,
    FrozenSetT_co,
    IterableT,
    IterableT_co,
    IteratorT,
    IteratorT_co,
    ListT,
    ListT_co,
    MappingTypes,
    NumericTypes,
    OptionalConverter,
    ParentSpecType,
    ParentSpecType_co,
    ParentSpecType_contra,
    PredicateFunc,
    PredicateFunc_contra,
    PrimitiveTypes,
    SchemaTypeT,
    SchemaTypeT_co,
    SchemaTypeT_contra,
    SchemaValueT,
    SchemaValueT_co,
    SchemaValueT_contra,
    SequenceT,
    SequenceT_co,
    SequenceTypes,
    SetT,
    SetT_co,
    SetTypes,
    TransformFunc,
    TransformFunc_co_contra,
    TryResult,
    TupleT,
    TupleT_co,
    TypeConverter,
    TypeConverterSafe,
    TypeDistance,
    TypeGuardFunc,
    TypeGuardFuncT,
    TypeHierarchy,
    TypeIdentifier,
    TypeMap,
    TypeMapFrom,
    TypeMapSR,
    TypeMapTo,
    TypeMatch,
    TypeName,
    TypePath,
    TypePrecedence,
    TypeRegistry,
    TypeRegistryT,
    TypeRegistryT_co,
    TypeRegistryT_contra,
    TypeRelationship,
    ValidationContext,
    ValidationFunc,
    ValidationFuncT,
    ValidationFuncT_contra,
    ValidationOptions,
    ValidationPath,
    ValidationResult,
    ValidationResultT,
    ValidationStrategy,
    ValidationWithPath,
)

# ===============================================================================
# Type Analysis - Relationship determination and compatibility analysis
# ===============================================================================
from type_forge.typing.analysis import TypeRelationshipAnalyzer

# ===============================================================================
# Type Conversion - Transformation utilities with elegant error handling
# ===============================================================================
from type_forge.typing.conversion import (
    ConversionResult,
    coerce_to_type,
    convert_with_fallback,
    safe_bool_convert,
    safe_float_convert,
    safe_int_convert,
    safe_str_convert,
    try_convert,
)

# ===============================================================================
# Type Definitions - Core enumerations and structural definitions
# ===============================================================================
from type_forge.typing.definitions import (
    TypeCategory,
    TypeCompatibility,
    ValidationLevel,
    ValidationSeverity,
)

# ===============================================================================
# Type Hints - Advanced hints for complex structures and schemas
# ===============================================================================
from type_forge.typing.hints import (
    CollectionT,
    ListSchemaT,
    PathSegmentT,
    PathT,
    SchemaNodeT,
    SchemaSequenceT,
    SchemaTypeMappingT,
    SchemaValueNodeT,
    SingleTypeT,
    UnionTypeT,
)

# ===============================================================================
# Type Mapping - Classification and relationship taxonomy
# ===============================================================================
from type_forge.typing.mapping import (
    describe_type,
    get_common_supertype,
    get_python_type_for_name,
    get_type_category,
    get_type_name,
)

# ===============================================================================
# Type Naming - Standardized naming conventions and utilities
# ===============================================================================
from type_forge.typing.naming import get_type_name as get_standardized_type_name
from type_forge.typing.naming import is_primitive_type

# ===============================================================================
# Type Protocols - Interface definitions for type behaviors
# ===============================================================================
from type_forge.typing.protocols import (
    CompositeValidator,
    SupportsBoolConversion,
    SupportsComparison,
    SupportsEquality,
    SupportsFloat,
    SupportsFloatConversion,
    SupportsGetAttr,
    SupportsGetItem,
    SupportsInt,
    SupportsIntConversion,
    SupportsIteration,
    SupportsLen,
    SupportsLength,
    SupportsMapping,
    SupportsStrConversion,
    SupportsTypeCheck,
)
from type_forge.typing.protocols import TypeConverter as TypeConverterProtocol
from type_forge.typing.protocols import TypedConverter, TypeDeduplicator, TypeFactory
from type_forge.typing.protocols import TypeForge
from type_forge.typing.protocols import TypeForge as TypeForgeProtocol
from type_forge.typing.protocols import TypeInfo, TypeNormalizer
from type_forge.typing.protocols import TypeRegistry as TypeRegistryProtocol
from type_forge.typing.protocols import TypeStandardizer, Validator

# ===============================================================================
# Type Standardization - Normalization and consistency utilities
# ===============================================================================
from type_forge.typing.standardization import (
    deduplicate_types,
    get_type_hierarchy,
    is_abstract_type,
    is_generic_type,
    standardize_type_name,
)

# ===============================================================================
# Type Validation - Verification utilities with configurable severity
# ===============================================================================
from type_forge.typing.validation import (
    ValidationIssue,
    ValidationReport,
    has_attributes,
    is_callable,
    is_collection,
    is_compatible_with_type,
    is_function,
    is_instance_of_any,
    is_method,
    is_non_empty_string,
    is_numeric,
    is_protocol_instance,
    is_subclass_safe,
    is_valid_identifier,
)

# ===============================================================================
# Type Variables - Generic variables with variance annotations
# ===============================================================================
from type_forge.typing.variables import (
    ComparableT,
    ComparableT_co,
    ComparableT_contra,
    HashableT,
    HashableT_co,
    HashableT_contra,
    K,
    K_co,
    K_contra,
    R,
    R_co,
    R_contra,
    S,
    S_co,
    S_contra,
    T,
    T_co,
    T_contra,
    TCallable,
    TCallable_co,
    TCallable_contra,
    TCollection,
    TCollection_co,
    TCollection_contra,
    TError,
    TError_co,
    TError_contra,
    TInstance,
    TInstance_co,
    TInstance_contra,
    TValue,
    TValue_co,
    TValue_contra,
    U,
    U_co,
    U_contra,
    V,
    V_co,
    V_contra,
)

# Complete and alphabetically sorted __all__ list for precise export control
__all__: list[str] = [
    # Type aliases
    "CollectionTypes",
    "ConverterMap",
    "ConverterMapGeneric",
    "ConverterMapSR",
    "ConverterPriority",
    "DictKV",
    "DictKV_co",
    "DictSchemaT",
    "ErrorHandler",
    "ErrorMessage",
    "FallbackProvider",
    "FieldDefinitions",
    "FieldDefinitionsT",
    "FieldDefinitionsT_co",
    "FieldDefinitionsT_contra",
    "FieldsWithDefaults",
    "FieldsWithDefaultsT",
    "FrozenSetT",
    "FrozenSetT_co",
    "IterableT",
    "IterableT_co",
    "IteratorT",
    "IteratorT_co",
    "ListT",
    "ListT_co",
    "MappingTypes",
    "NumericTypes",
    "OptionalConverter",
    "ParentSpecType",
    "ParentSpecType_co",
    "ParentSpecType_contra",
    "PredicateFunc",
    "PredicateFunc_contra",
    "PrimitiveTypes",
    "SchemaTypeT",
    "SchemaTypeT_co",
    "SchemaTypeT_contra",
    "SchemaValueT",
    "SequenceT",
    "SequenceT_co",
    "SequenceTypes",
    "SetT",
    "SetT_co",
    "SetTypes",
    "TransformFunc",
    "TransformFunc_co_contra",
    "TryResult",
    "TupleT",
    "TupleT_co",
    "TypeConverter",
    "TypeConverterSafe",
    "TypeDistance",
    "TypeGuardFunc",
    "TypeGuardFuncT",
    "TypeHierarchy",
    "TypeIdentifier",
    "TypeMap",
    "TypeMapFrom",
    "TypeMapSR",
    "TypeMapTo",
    "TypeMatch",
    "TypeName",
    "TypePath",
    "TypePrecedence",
    "TypeRegistry",
    "TypeRegistryT",
    "TypeRegistryT_co",
    "TypeRegistryT_contra",
    "TypeRelationship",
    "ValidationContext",
    "ValidationFunc",
    "ValidationFuncT",
    "ValidationFuncT_contra",
    "ValidationOptions",
    "ValidationPath",
    "ValidationResult",
    "ValidationResultT",
    "ValidationStrategy",
    "ValidationWithPath",
    # Type analysis
    "TypeRelationshipAnalyzer",
    # Type conversion utilities
    "ConversionResult",
    "coerce_to_type",
    "convert_with_fallback",
    "safe_bool_convert",
    "safe_float_convert",
    "safe_int_convert",
    "safe_str_convert",
    "try_convert",
    # Type definitions and enumerations
    "TypeCategory",
    "TypeCompatibility",
    "ValidationLevel",
    "ValidationSeverity",
    # Type hints
    "CollectionT",
    "ListSchemaT",
    "PathSegmentT",
    "PathT",
    "SchemaNodeT",
    "SchemaSequenceT",
    "SchemaTypeMappingT",
    "SchemaValueNodeT",
    "SchemaValueT",
    "SingleTypeT",
    "UnionTypeT",
    # Type mapping and classification
    "describe_type",
    "get_common_supertype",
    "get_python_type_for_name",
    "get_type_category",
    "get_type_name",
    # Type naming utilities
    "get_standardized_type_name",
    "is_primitive_type",
    # Type protocols and interfaces
    "CompositeValidator",
    "SupportsBoolConversion",
    "SupportsComparison",
    "SupportsEquality",
    "SupportsFloat",
    "SupportsFloatConversion",
    "SupportsGetAttr",
    "SupportsGetItem",
    "SupportsInt",
    "SupportsIntConversion",
    "SupportsIteration",
    "SupportsLen",
    "SupportsLength",
    "SupportsMapping",
    "SupportsStrConversion",
    "SupportsTypeCheck",
    "TypeConverterProtocol",
    "TypeDeduplicator",
    "TypedConverter",
    "TypeFactory",
    "TypeForge",
    "TypeInfo",
    "TypeNormalizer",
    "TypeRegistryProtocol",
    "TypeStandardizer",
    "Validator",
    "TypeForgeProtocol",
    # Type standardization
    "deduplicate_types",
    "get_type_hierarchy",
    "is_abstract_type",
    "is_generic_type",
    "standardize_type_name",
    # Type validation utilities
    "ValidationIssue",
    "ValidationReport",
    "has_attributes",
    "is_callable",
    "is_collection",
    "is_compatible_with_type",
    "is_function",
    "is_instance_of_any",
    "is_method",
    "is_non_empty_string",
    "is_numeric",
    "is_protocol_instance",
    "is_subclass_safe",
    "is_valid_identifier",
    # Type variables with variance annotations
    "ComparableT",
    "ComparableT_co",
    "ComparableT_contra",
    "HashableT",
    "HashableT_co",
    "HashableT_contra",
    "K",
    "K_co",
    "K_contra",
    "R",
    "R_co",
    "R_contra",
    "S",
    "S_co",
    "S_contra",
    "T",
    "TCallable",
    "TCallable_co",
    "TCallable_contra",
    "TCollection",
    "TCollection_co",
    "TCollection_contra",
    "TError",
    "TError_co",
    "TError_contra",
    "TInstance",
    "TInstance_co",
    "TInstance_contra",
    "TValue",
    "TValue_co",
    "TValue_contra",
    "T_co",
    "T_contra",
    "U",
    "U_co",
    "U_contra",
    "V",
    "V_co",
    "V_contra",
    "SchemaValueT_co",
    "SchemaValueT_contra",
    "DictSchemaT_co",
    "DictSchemaT_contra",
]

# Module version with semantic versioning
__version__: str = "0.1.0"
version: str = __version__

# Author Information
__author__: str = "Lloyd Handyside"
author: str = __author__


def _verify_exports() -> None:
    """Verify that all items in __all__ are actually defined in this module.

    This function performs a recursive self-check to ensure the integrity of exports,
    maintaining the Eidosian principle of recursive refinement and quality assurance.

    Raises:
        ImportError: If any item in __all__ is not defined in the module
    """
    module = sys.modules[__name__]
    missing_items: list[str] = [name for name in __all__ if not hasattr(module, name)]

    if missing_items:
        missing_str = ", ".join(missing_items)
        raise ImportError(
            f"Items in __all__ not defined in {__name__}: {missing_str}",
        )


# Run verification if not importing during module load
if not hasattr(sys, "_called_from_test") and __name__ == "__main__":
    _verify_exports()
