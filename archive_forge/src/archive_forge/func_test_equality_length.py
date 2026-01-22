import re
import unittest
from oslo_config import types
def test_equality_length(self):
    a = types.URI(max_length=5)
    b = types.URI(max_length=5)
    self.assertEqual(a, b)