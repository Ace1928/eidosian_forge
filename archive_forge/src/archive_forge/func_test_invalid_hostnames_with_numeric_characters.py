import re
import unittest
from oslo_config import types
def test_invalid_hostnames_with_numeric_characters(self):
    self.assertInvalid('10.0.0.0')
    self.assertInvalid('3.14')
    self.assertInvalid('___site0.1001')
    self.assertInvalid('org.10')
    self.assertInvalid('0.0.00')