import re
import unittest
from oslo_config import types
def test_equality_schemes_not(self):
    a = types.URI()
    b = types.URI(schemes=['ftp'])
    c = types.URI(schemes=['http'])
    self.assertNotEqual(a, b)
    self.assertNotEqual(c, b)