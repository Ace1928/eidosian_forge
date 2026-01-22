import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_flat_empty(self):

    def check(arr, arrty):
        cfunc = njit((arrty,))(array_flat_sum)
        cres = cfunc.overloads[arrty,]
        got = cres.entry_point(arr)
        expected = cfunc.py_func(arr)
        self.assertPreciseEqual(expected, got)
    arr = np.zeros(0, dtype=np.int32)
    arr = arr.reshape(0, 2)
    arrty = types.Array(types.int32, 2, layout='C')
    check(arr, arrty)
    arrty = types.Array(types.int32, 2, layout='F')
    check(arr, arrty)
    arrty = types.Array(types.int32, 2, layout='A')
    check(arr, arrty)
    arr = arr.reshape(2, 0)
    arrty = types.Array(types.int32, 2, layout='C')
    check(arr, arrty)
    arrty = types.Array(types.int32, 2, layout='F')
    check(arr, arrty)
    arrty = types.Array(types.int32, 2, layout='A')
    check(arr, arrty)