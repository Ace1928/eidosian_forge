import pytest
from type_forge import TypeCore, ValidationError

def test_basic_validation():
    tc = TypeCore()
    schema = {"type": "string", "minLength": 2}
    assert tc.check_schema(schema, "Hello")
    
    with pytest.raises(ValidationError):
        tc.check_schema(schema, "H")

def test_pydantic_acceleration():
    tc = TypeCore()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    tc.register_schema("Person", schema)
    
    # Should use Pydantic backend
    assert tc.validate("Person", {"name": "Alice", "age": 30})
    
    with pytest.raises(ValidationError):
        tc.validate("Person", {"age": 30}) # Missing name
