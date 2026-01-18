"""
Type Forge - Schema registry and validation.
Standardizes communication between Eidosian components.
"""
from typing import Any, Dict, List, Optional, Type, Union
import re

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class TypeCore:
    """
    Registry for component schemas and validation logic.
    Supports basic types, regex patterns, ranges, and nested structures.
    """
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a schema definition."""
        self._schemas[name] = schema

    def validate(self, name: str, data: Any) -> bool:
        """Validate data against a registered schema."""
        if name not in self._schemas:
            raise ValueError(f"Schema '{name}' not found.")
        
        schema = self._schemas[name]
        return self._validate_recursive(schema, data)

    def check_schema(self, schema: Dict[str, Any], data: Any) -> bool:
        """Validate data against an ad-hoc schema."""
        return self._validate_recursive(schema, data)

    def _validate_recursive(self, schema: Dict[str, Any], data: Any) -> bool:
        # Basic type checking
        expected_type = schema.get("type")
        if expected_type == "string":
            if not isinstance(data, str):
                raise ValidationError(f"Expected string, got {type(data).__name__}")
            if "pattern" in schema and not re.match(schema["pattern"], data):
                raise ValidationError(f"String does not match pattern: {schema['pattern']}")
            if "minLength" in schema and len(data) < schema["minLength"]:
                raise ValidationError(f"String too short (min {schema['minLength']})")
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                raise ValidationError(f"String too long (max {schema['maxLength']})")
                
        elif expected_type == "number":
            if not isinstance(data, (int, float)):
                raise ValidationError(f"Expected number, got {type(data).__name__}")
            if "minimum" in schema and data < schema["minimum"]:
                raise ValidationError(f"Value too small (min {schema['minimum']})")
            if "maximum" in schema and data > schema["maximum"]:
                raise ValidationError(f"Value too large (max {schema['maximum']})")
                
        elif expected_type == "boolean":
            if not isinstance(data, bool):
                raise ValidationError(f"Expected boolean, got {type(data).__name__}")
                
        elif expected_type == "array":
            if not isinstance(data, list):
                raise ValidationError(f"Expected array, got {type(data).__name__}")
            if "minItems" in schema and len(data) < schema["minItems"]:
                raise ValidationError(f"Array too short (min {schema['minItems']})")
            if "items" in schema:
                for item in data:
                    self._validate_recursive(schema["items"], item)
                    
        elif expected_type == "object":
            if not isinstance(data, dict):
                raise ValidationError(f"Expected object, got {type(data).__name__}")
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for req in required:
                if req not in data:
                    raise ValidationError(f"Missing required property: {req}")
            
            for key, value in data.items():
                if key in properties:
                    self._validate_recursive(properties[key], value)
                elif not schema.get("additionalProperties", True):
                    raise ValidationError(f"Additional property not allowed: {key}")

        elif expected_type == "any":
            pass # Allow anything
            
        return True
