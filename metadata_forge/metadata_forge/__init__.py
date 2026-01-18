"""
🔮 Eidosian Metadata Forge

A universal metadata system for intelligence amplification through structured
documentation, introspection, and self-optimization of software components.

This package provides tools for defining, validating, and working with 
comprehensive metadata that transforms code components into self-aware,
fully documented, and universally linkable entities.
"""

__version__ = "0.1.0"

from .schema import (
    EntityMetadata,
    MetadataField,
    SchemaDefinition,
    create_metadata_template,
    validate_metadata
)

__all__ = [
    # Core schema components
    "EntityMetadata",
    "MetadataField",
    "SchemaDefinition",
    "create_metadata_template",
    "validate_metadata",
]
