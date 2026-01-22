import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
def test_assign_const_byte_string(self):

    @cuda.jit
    def bytes_assign(arr):
        i = cuda.grid(1)
        if i < len(arr):
            arr[i] = b'XYZ'
    n_strings = 8
    arr = np.zeros(n_strings + 1, dtype='S12')
    bytes_assign[1, n_strings](arr)
    expected = np.zeros_like(arr)
    expected[:-1] = b'XYZ'
    expected[-1] = b''
    np.testing.assert_equal(arr, expected)