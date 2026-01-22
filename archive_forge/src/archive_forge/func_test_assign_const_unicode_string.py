import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
def test_assign_const_unicode_string(self):

    @cuda.jit
    def str_assign(arr):
        i = cuda.grid(1)
        if i < len(arr):
            arr[i] = 'XYZ'
    n_strings = 8
    arr = np.zeros(n_strings + 1, dtype='<U12')
    str_assign[1, n_strings](arr)
    expected = np.zeros_like(arr)
    expected[:-1] = 'XYZ'
    expected[-1] = ''
    np.testing.assert_equal(arr, expected)