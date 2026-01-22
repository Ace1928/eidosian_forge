import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_int16_min_value(self):
    props = {'key': int16_t(-2 ** 15)}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntDictsEqual(props, res)