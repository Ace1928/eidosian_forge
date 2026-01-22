from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_argmax_axis_1d_2d_4d(self):
    arr1d = np.array([0, 20, 3, 4])
    arr2d = np.arange(6).reshape(2, 3)
    arr2d[0, 1] += 100
    arr4d = np.arange(120).reshape(2, 3, 4, 5) + 10
    arr4d[0, 1, 1, 2] += 100
    arr4d[1, 0, 0, 0] -= 51
    for arr in [arr1d, arr2d, arr4d]:
        axes = list(range(arr.ndim)) + [-(i + 1) for i in range(arr.ndim)]
        py_functions = [lambda a, _axis=axis: np.argmax(a, axis=_axis) for axis in axes]
        c_functions = [jit(nopython=True)(pyfunc) for pyfunc in py_functions]
        for cfunc in c_functions:
            self.assertPreciseEqual(cfunc.py_func(arr), cfunc(arr))