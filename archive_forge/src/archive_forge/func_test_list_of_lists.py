import re
import unittest
from oslo_config import types
def test_list_of_lists(self):
    self.type_instance = types.List(types.List(types.String(), bounds=True))
    self.assertConvertedValue('[foo],[bar, baz],[bam]', [['foo'], ['bar', 'baz'], ['bam']])