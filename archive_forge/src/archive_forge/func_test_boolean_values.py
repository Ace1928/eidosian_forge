import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_boolean_values(self):
    props = {'key1': True, 'key2': False}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)