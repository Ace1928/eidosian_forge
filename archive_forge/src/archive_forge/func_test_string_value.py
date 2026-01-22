import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_string_value(self):
    props = {'key': 'value'}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)