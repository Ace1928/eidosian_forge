import re
import unittest
from oslo_config import types
def test_list_of_values_containing_commas(self):
    self.type_instance = types.List(types.String(quotes=True))
    self.assertConvertedValue('foo,"bar, baz",bam', ['foo', 'bar, baz', 'bam'])