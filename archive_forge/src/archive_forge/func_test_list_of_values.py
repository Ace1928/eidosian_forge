import re
import unittest
from oslo_config import types
def test_list_of_values(self):
    self.assertConvertedValue(' foo bar, baz ', ['foo bar', 'baz'])