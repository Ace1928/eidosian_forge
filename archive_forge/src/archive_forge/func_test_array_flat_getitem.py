import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_flat_getitem(self):
    pyfunc = array_flat_getitem
    cfunc = njit(pyfunc)

    def check(arr, ind):
        expected = pyfunc(arr, ind)
        self.assertEqual(cfunc(arr, ind), expected)
    arr = np.arange(24).reshape(4, 2, 3)
    for i in range(arr.size):
        check(arr, i)
    arr = arr.T
    for i in range(arr.size):
        check(arr, i)
    arr = arr[::2]
    for i in range(arr.size):
        check(arr, i)
    arr = np.array([42]).reshape(())
    for i in range(arr.size):
        check(arr, i)
    arr = np.bool_([1, 0, 0, 1])
    for i in range(arr.size):
        check(arr, i)
    arr = arr[::2]
    for i in range(arr.size):
        check(arr, i)