import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_implicit_uint32_max_value(self):
    props = {'rewind-request': 2 ** 32 - 1}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntDictsEqual(props, res)