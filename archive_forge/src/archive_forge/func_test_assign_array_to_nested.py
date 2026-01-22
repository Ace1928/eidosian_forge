import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_assign_array_to_nested(self):
    src = (np.arange(3) + 1).astype(np.int16)
    got = np.zeros(2, dtype=nested_array1_dtype)
    expected = np.zeros(2, dtype=nested_array1_dtype)
    pyfunc = assign_array_to_nested
    kernel = cuda.jit(pyfunc)
    kernel[1, 1](got[0], src)
    pyfunc(expected[0], src)
    np.testing.assert_array_equal(expected, got)