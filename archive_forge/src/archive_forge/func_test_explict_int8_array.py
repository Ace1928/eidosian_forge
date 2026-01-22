import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_int8_array(self):
    props = {'key': [int8_t(0), int8_t(1), int8_t(2 ** 7 - 1), int8_t(-2 ** 7)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)