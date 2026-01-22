from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_it_catches_registered_errors(self):
    checker = FormatChecker()
    checker.checks('boom', raises=type(BOOM))(boom)
    with self.assertRaises(FormatError) as cm:
        checker.check(instance=12, format='boom')
    self.assertIs(cm.exception.cause, BOOM)
    self.assertIs(cm.exception.__cause__, BOOM)
    self.assertEqual(str(cm.exception), "12 is not a 'boom'")
    with self.assertRaises(type(BANG)):
        checker.check(instance='bang', format='boom')