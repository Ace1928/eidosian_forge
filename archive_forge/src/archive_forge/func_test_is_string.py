import unittest
from aniso8601.compat import PY2, is_string
def test_is_string(self):
    self.assertTrue(is_string('asdf'))
    self.assertTrue(is_string(''))
    if PY2 is True:
        self.assertTrue(is_string(unicode('asdf')))
    self.assertFalse(is_string(None))
    self.assertFalse(is_string(123))
    self.assertFalse(is_string(4.56))
    self.assertFalse(is_string([]))
    self.assertFalse(is_string({}))