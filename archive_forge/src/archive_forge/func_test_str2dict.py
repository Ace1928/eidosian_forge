import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_str2dict(self):
    string = 'key1=value1,key2=value2,key3=value3'
    expected = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    self.assertEqual(expected, helpers.str2dict(string))