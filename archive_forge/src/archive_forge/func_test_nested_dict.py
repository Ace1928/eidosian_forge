import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_nested_dict(self):
    props = {'key': {}}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)