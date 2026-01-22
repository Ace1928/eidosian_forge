from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_it_can_register_checkers(self):
    checker = FormatChecker()
    checker.checks('boom')(boom)
    self.assertEqual(checker.checkers, dict(FormatChecker.checkers, boom=(boom, ())))