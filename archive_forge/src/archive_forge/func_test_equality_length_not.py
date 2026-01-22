import re
import unittest
from oslo_config import types
def test_equality_length_not(self):
    a = types.URI()
    b = types.URI(max_length=5)
    c = types.URI(max_length=10)
    self.assertNotEqual(a, b)
    self.assertNotEqual(c, b)