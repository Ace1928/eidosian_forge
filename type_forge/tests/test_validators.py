import unittest
from type_forge.validators.basic import BasicValidator
from type_forge.validators.composite import CompositeValidator

class TestBasicValidator(unittest.TestCase):
    def setUp(self):
        self.validator = BasicValidator()

    def test_validate_integer(self):
        self.assertTrue(self.validator.validate(10, int))
        self.assertFalse(self.validator.validate("10", int))

    def test_validate_string(self):
        self.assertTrue(self.validator.validate("hello", str))
        self.assertFalse(self.validator.validate(10, str))

class TestCompositeValidator(unittest.TestCase):
    def setUp(self):
        self.basic_validator = BasicValidator()
        self.composite_validator = CompositeValidator([self.basic_validator])

    def test_validate_composite(self):
        self.assertTrue(self.composite_validator.validate(10, int))
        self.assertFalse(self.composite_validator.validate("10", int))
        self.assertTrue(self.composite_validator.validate("hello", str))
        self.assertFalse(self.composite_validator.validate(10, str))

if __name__ == '__main__':
    unittest.main()