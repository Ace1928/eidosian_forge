import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_issue_7693(self):
    src_dtype = np.dtype([('user', np.float64), ('array', np.int16, (3,))], align=True)
    dest_dtype = np.dtype([('user1', np.float64), ('array1', np.int16, (3,))], align=True)

    @cuda.jit
    def copy(index, src, dest):
        dest['user1'] = src[index]['user']
        dest['array1'] = src[index]['array']
    source = np.zeros(2, dtype=src_dtype)
    got = np.zeros(2, dtype=dest_dtype)
    expected = np.zeros(2, dtype=dest_dtype)
    source[0] = (1.2, [1, 2, 3])
    copy[1, 1](0, source, got[0])
    copy.py_func(0, source, expected[0])
    np.testing.assert_array_equal(expected, got)