import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_mismatching_values_array2(self):
    props = {'key': [True, 10]}
    with self.assertRaises(TypeError):
        self._dict_to_nvlist_to_dict(props)