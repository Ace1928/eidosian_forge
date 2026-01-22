import re
import unittest
from oslo_config import types
def test_equality_schemes(self):
    a = types.URI(schemes=['ftp'])
    b = types.URI(schemes=['ftp'])
    self.assertEqual(a, b)