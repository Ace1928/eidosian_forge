import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_implicit_int32_max_value(self):
    props = {'pool_context': 2 ** 31 - 1}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntDictsEqual(props, res)