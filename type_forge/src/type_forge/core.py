from eidosian_core import eidosian
"""
Type Forge - Schema registry and validation.
Standardizes communication between Eidosian components.
"""
from typing import Any, Dict, Optional, Protocol, Type
import re
from pydantic import BaseModel, ValidationError as PydanticValidationError, create_model


class RegistryBackend(Protocol):
    """Minimal persistence backend contract."""

    def get(self, key: str, default: Any = None, use_env: bool = True) -> Any: ...

    def set(self, key: str, value: Any, notify: bool = True, persist: bool = True) -> None: ...


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class TypeCore:
    """
    Registry for component schemas and validation logic.
    Supports basic types, regex patterns, ranges, and nested structures.
    Now supports Pydantic dynamic models.
    """
    
    def __init__(
        self,
        registry_backend: RegistryBackend | None = None,
        registry_key: str = "type_forge.schemas",
    ):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, Type[BaseModel]] = {}
        self._registry_backend = registry_backend
        self._registry_key = registry_key
        self._load_from_backend()

    def _load_from_backend(self) -> None:
        if self._registry_backend is None:
            return
        try:
            snapshot = self._registry_backend.get(self._registry_key, default={})
            if isinstance(snapshot, dict):
                self.restore(snapshot)
        except Exception:
            # Backend connectivity should never prevent core operation.
            return

    def _persist_to_backend(self) -> None:
        if self._registry_backend is None:
            return
        try:
            self._registry_backend.set(self._registry_key, self.snapshot())
        except Exception:
            return

    @eidosian()
    def register_schema(self, name: str, schema: Dict[str, Any]) -> bool:
        """Register a schema definition. Returns True if updated."""
        if self._schemas.get(name) == schema:
            return False
        self._schemas[name] = schema
        # Attempt to compile to Pydantic model for performance
        try:
            self._models[name] = self._compile_to_pydantic(name, schema)
        except Exception:
            self._models.pop(name, None)
        self._persist_to_backend()
        return True

    @eidosian()
    def delete_schema(self, name: str) -> bool:
        """Delete a registered schema definition."""
        if name not in self._schemas:
            return False
        del self._schemas[name]
        self._models.pop(name, None)
        self._persist_to_backend()
        return True

    @eidosian()
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered schema definition."""
        return self._schemas.get(name)

    @eidosian()
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of registered schemas."""
        return dict(self._schemas)

    @eidosian()
    def restore(self, snapshot: Dict[str, Dict[str, Any]]) -> None:
        """Restore registered schemas from a snapshot."""
        self._schemas = dict(snapshot)
        self._models = {}
        for name, schema in self._schemas.items():
            try:
                self._models[name] = self._compile_to_pydantic(name, schema)
            except Exception:
                continue
        self._persist_to_backend()

    @eidosian()
    def validate(self, name: str, data: Any) -> bool:
        """Validate data against a registered schema."""
        if name in self._models:
            try:
                self._models[name](**data)
                return True
            except PydanticValidationError as e:
                raise ValidationError(f"Pydantic validation failed: {e}")
        
        if name not in self._schemas:
            raise ValueError(f"Schema '{name}' not found.")
        
        schema = self._schemas[name]
        return self._validate_recursive(schema, data)

    @eidosian()
    def check_schema(self, schema: Dict[str, Any], data: Any) -> bool:
        """Validate data against an ad-hoc schema."""
        return self._validate_recursive(schema, data)

    def _compile_to_pydantic(self, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Rough conversion of JSON-like schema to Pydantic model."""
        fields = {}
        if schema.get("type") != "object":
            raise ValueError("Only object schemas supported for Pydantic conversion")
            
        for prop, prop_schema in schema.get("properties", {}).items():
            t = str
            if prop_schema.get("type") == "number": t = float
            elif prop_schema.get("type") == "integer": t = int
            elif prop_schema.get("type") == "boolean": t = bool
            
            is_required = prop in schema.get("required", [])
            default = ... if is_required else None
            fields[prop] = (t, default)
            
        return create_model(name, **fields)

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
                
        elif expected_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                raise ValidationError(f"Expected integer, got {type(data).__name__}")
            if "minimum" in schema and data < schema["minimum"]:
                raise ValidationError(f"Value too small (min {schema['minimum']})")
            if "maximum" in schema and data > schema["maximum"]:
                raise ValidationError(f"Value too large (max {schema['maximum']})")
                
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
