import re
import unittest
from wsme import exc
from wsme import types
def test_validate_integer_type(self):
    v = types.IntegerType(minimum=1, maximum=10)
    v.validate(1)
    v.validate(5)
    v.validate(10)
    self.assertRaises(ValueError, v.validate, 0)
    self.assertRaises(ValueError, v.validate, 11)