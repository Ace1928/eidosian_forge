import re
import unittest
from oslo_config import types
def test_with_max_and_min(self):
    t = types.Port(min=123, max=456)
    self.assertRaises(ValueError, t, 122)
    t(123)
    t(300)
    t(456)
    self.assertRaises(ValueError, t, 0)
    self.assertRaises(ValueError, t, 457)