import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_implicit_uint32_negative_value(self):
    with self.assertRaises(OverflowError):
        props = {'rewind-request': -1}
        self._dict_to_nvlist_to_dict(props)