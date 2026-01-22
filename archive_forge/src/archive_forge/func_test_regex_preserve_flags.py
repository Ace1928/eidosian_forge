import re
import unittest
from oslo_config import types
def test_regex_preserve_flags(self):
    self.type_instance = types.String(regex=re.compile('^[A-Z]', re.I), ignore_case=False)
    self.assertConvertedValue('foo', 'foo')