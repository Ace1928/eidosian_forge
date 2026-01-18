"""
Tests for metadata schema validation.
"""
import pytest
from metadata_forge.schema import (
    create_metadata_template,
    validate_metadata,
    EntityMetadata
)


def test_create_metadata_template():
    """Test creation of metadata template."""
    template = create_metadata_template()
    assert "entity" in template
    assert "identifier" in template
    assert "version" in template
    assert "purpose" in template
    assert "context" in template
    assert "parameters" in template
    assert "returns" in template
    assert "test_verification" in template
    assert "performance_profile" in template
    assert "behavioral_notes" in template
    assert "interactions" in template
    assert "modifiability" in template
    assert "programmatic_traceability" in template


def test_validate_metadata_valid():
    """Test validation of valid metadata."""
    metadata = {
        "entity": "Function",
        "identifier": "test_function",
        "version": "1.0.0",
        "purpose": "Test function for validation",
        "context": "Used in unit tests",
        "parameters": [
            {
                "name": "param1",
                "type": "str",
                "optional": False,
                "default": None,
                "description": "First parameter"
            }
        ],
        "returns": {
            "type": "bool",
            "description": "Success status"
        }
    }
    errors = validate_metadata(metadata)
    assert errors == {}


def test_validate_metadata_missing_required():
    """Test validation with missing required fields."""
    metadata = {
        "entity": "Function",
        "identifier": "test_function",
        # Missing version
        "purpose": "Test function",
        # Missing context
        "parameters": [],
        "returns": {
            "type": "bool",
            "description": "Success status"
        }
    }
    errors = validate_metadata(metadata)
    assert "context" in errors
    assert "version" in errors


def test_validate_metadata_invalid_entity():
    """Test validation with invalid entity type."""
    metadata = {
        "entity": "InvalidType",
        "identifier": "test_function",
        "version": "1.0.0",
        "purpose": "Test function",
        "context": "Used in tests",
        "parameters": [],
        "returns": {
            "type": "bool",
            "description": "Success status"
        }
    }
    errors = validate_metadata(metadata)
    assert "entity" in errors
