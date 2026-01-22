import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_nested_gen(self):

    def gen0(arr):
        for i in range(arr.size):
            yield arr

    def factory(gen0):

        def gen1(arr):
            out = np.zeros_like(arr)
            for x in gen0(arr):
                out = out + x
            return (out, arr)
        return gen1
    py_arr = np.arange(10)
    c_arr = py_arr.copy()
    py_res, py_old = factory(gen0)(py_arr)
    c_gen = jit(nopython=True)(factory(jit(nopython=True)(gen0)))
    c_res, c_old = c_gen(c_arr)
    self.assertIsNot(py_arr, c_arr)
    self.assertIs(py_old, py_arr)
    self.assertIs(c_old, c_arr)
    np.testing.assert_equal(py_res, c_res)
    self.assertRefCountEqual(py_res, c_res)