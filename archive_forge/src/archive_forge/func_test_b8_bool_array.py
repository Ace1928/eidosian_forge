from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_b8_bool_array(self):
    arr1 = np.array([((True, True, False),), ((True, False, True),)], dtype=np.dtype([('x', ('?', (3,)))]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))]), cast_dtype=np.dtype([('x', ('?', (3,)))]))