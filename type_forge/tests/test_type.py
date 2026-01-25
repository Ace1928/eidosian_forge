import unittest
from type_forge import TypeCore, ValidationError

class TestTypeCore(unittest.TestCase):
    def setUp(self):
        self.types = TypeCore()

    def test_basic_validation(self):
        schema = {"type": "string"}
        self.types.register_schema("my_string", schema)
        self.assertTrue(self.types.validate("my_string", "hello"))
        
        with self.assertRaises(ValidationError):
            self.types.validate("my_string", 123)

    def test_object_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }
        self.types.register_schema("person", schema)
        
        self.assertTrue(self.types.validate("person", {"name": "Lloyd", "age": 30}))
        self.assertTrue(self.types.validate("person", {"name": "Lloyd"}))
        
        with self.assertRaises(ValidationError):
            self.types.validate("person", {"age": 30}) # Missing name

    def test_pattern_validation(self):
        schema = {"type": "string", "pattern": r"^v\d+\.\d+\.\d+$"}
        self.assertTrue(self.types.check_schema(schema, "v1.0.0"))
        with self.assertRaises(ValidationError):
            self.types.check_schema(schema, "1.0.0")

    def test_range_validation(self):
        schema = {"type": "number", "minimum": 0, "maximum": 100}
        self.assertTrue(self.types.check_schema(schema, 50))
        with self.assertRaises(ValidationError):
            self.types.check_schema(schema, 101)

    def test_length_validation(self):
        schema = {"type": "string", "minLength": 3, "maxLength": 5}
        self.assertTrue(self.types.check_schema(schema, "abc"))
        with self.assertRaises(ValidationError):
            self.types.check_schema(schema, "ab")
        with self.assertRaises(ValidationError):
            self.types.check_schema(schema, "abcdef")

if __name__ == "__main__":
    unittest.main()
