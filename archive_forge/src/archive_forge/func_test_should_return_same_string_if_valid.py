import re
import unittest
from oslo_config import types
def test_should_return_same_string_if_valid(self):
    self.assertConvertedValue('foo bar', 'foo bar')