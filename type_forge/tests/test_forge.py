"""Tests for TypeCore schema registration and validation."""
import unittest
from type_forge import TypeCore, ValidationError


class TestTypeForgeSchemas(unittest.TestCase):
    """Test TypeCore schema management and dynamic validation."""

    def setUp(self):
        self.core = TypeCore()

    def test_register_schema(self):
        """Test schema registration."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }
        result = self.core.register_schema("person", schema)
        self.assertTrue(result)
        
        # Re-registering same schema returns False
        result = self.core.register_schema("person", schema)
        self.assertFalse(result)

    def test_delete_schema(self):
        """Test schema deletion."""
        schema = {"type": "string"}
        self.core.register_schema("temp", schema)
        
        result = self.core.delete_schema("temp")
        self.assertTrue(result)
        
        # Deleting non-existent schema returns False
        result = self.core.delete_schema("nonexistent")
        self.assertFalse(result)

    def test_get_schema(self):
        """Test schema retrieval."""
        schema = {"type": "number", "minimum": 0}
        self.core.register_schema("positive", schema)
        
        retrieved = self.core.get_schema("positive")
        self.assertEqual(retrieved, schema)
        
        # Non-existent schema returns None
        self.assertIsNone(self.core.get_schema("missing"))

    def test_object_validation(self):
        """Test object validation with required fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }
        self.core.register_schema("person", schema)
        
        # Valid data
        self.assertTrue(self.core.validate("person", {"name": "Alice", "age": 30}))
        
        # Missing required field
        with self.assertRaises(ValidationError):
            self.core.validate("person", {"age": 30})


if __name__ == "__main__":
    unittest.main()
