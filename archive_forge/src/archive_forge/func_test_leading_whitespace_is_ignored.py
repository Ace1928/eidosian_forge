import re
import unittest
from oslo_config import types
def test_leading_whitespace_is_ignored(self):
    self.assertConvertedValue('   5', 5)