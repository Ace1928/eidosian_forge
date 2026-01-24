# 🔷 Type Forge Installation

Runtime type validation and schema management utilities.

## Quick Install

```bash
# Install
pip install -e ./type_forge

# Verify
python -c "from type_forge import TypeRegistry; print('✓ Ready')"
```

## CLI Usage

```bash
# Validate a schema
type-forge validate schema.json

# Check type compatibility
type-forge check-type '{"name": "test"}' UserSchema

# Help
type-forge --help
```

## Python API

```python
from type_forge import TypeRegistry, validate

registry = TypeRegistry()
registry.register_schema("User", {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
})

# Validate data
validate({"name": "Alice", "age": 30}, "User", registry)
```

## Dependencies

- `eidosian_core` - Universal decorators and logging

