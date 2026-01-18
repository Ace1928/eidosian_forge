"""
Metadata schema definition module for Eidosian systems.

This module provides the foundational schema structures that power
the universal metadata system for Eidosian software components.
"""

from .core import (
    EntityMetadata,
    MetadataField,
    SchemaDefinition,
    create_metadata_template,
    validate_metadata
)

__all__ = [
    "EntityMetadata",
    "MetadataField", 
    "SchemaDefinition",
    "create_metadata_template",
    "validate_metadata"
]
