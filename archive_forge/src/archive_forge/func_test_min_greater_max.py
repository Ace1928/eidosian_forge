import re
import unittest
from oslo_config import types
def test_min_greater_max(self):
    self.assertRaises(ValueError, types.Port, min=100, max=50)
    self.assertRaises(ValueError, types.Port, min=-50, max=-100)
    self.assertRaises(ValueError, types.Port, min=0, max=-50)
    self.assertRaises(ValueError, types.Port, min=50, max=0)