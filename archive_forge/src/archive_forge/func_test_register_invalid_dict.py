import re
import unittest
from wsme import exc
from wsme import types
def test_register_invalid_dict(self):
    self.assertRaises(ValueError, types.register_type, {})
    self.assertRaises(ValueError, types.register_type, {int: str, str: int})
    self.assertRaises(ValueError, types.register_type, {types.Unset: str})