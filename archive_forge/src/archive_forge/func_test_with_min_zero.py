import re
import unittest
from oslo_config import types
def test_with_min_zero(self):
    t = types.Port(min=0, max=456)
    self.assertRaises(ValueError, t, -1)
    t(0)
    t(123)
    t(300)
    t(456)
    self.assertRaises(ValueError, t, -201)
    self.assertRaises(ValueError, t, 457)