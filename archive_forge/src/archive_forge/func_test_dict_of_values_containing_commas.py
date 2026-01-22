import re
import unittest
from oslo_config import types
def test_dict_of_values_containing_commas(self):
    self.type_instance = types.Dict(types.String(quotes=True))
    self.assertConvertedValue('foo:"bar, baz",bam:quux', {'foo': 'bar, baz', 'bam': 'quux'})