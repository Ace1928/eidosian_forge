import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_issue_3158_2(self):
    dtype1 = np.dtype([('a', 'i8'), ('b', 'i4')])
    dtype2 = np.dtype((dtype1, (2, 2)))
    dtype3 = np.dtype([('x', '?'), ('y', dtype2)])

    @njit
    def fn(arr):
        return arr[0]
    arr = np.asarray([(False, [[(0, 1), (2, 3)], [(4, 5), (6, 7)]]), (True, [[(8, 9), (10, 11)], [(12, 13), (14, 15)]])], dtype=dtype3)
    expected = fn.py_func(arr)
    actual = fn(arr)
    self.assertEqual(expected, actual)