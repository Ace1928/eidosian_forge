import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_uint64_array(self):
    props = {'key': [uint64_t(0), uint64_t(1), uint64_t(2 ** 64 - 1)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)