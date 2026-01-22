from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_array_reshape(self):
    pyfuncs_to_use = [array_reshape, numpy_array_reshape]

    def generic_run(pyfunc, arr, shape):
        return pyfunc(arr, shape)

    @from_generic(pyfuncs_to_use)
    def check(pyfunc, arr, shape):
        expected = pyfunc.py_func(arr, shape)
        self.memory_leak_setup()
        got = generic_run(pyfunc, arr, shape)
        self.assertPreciseEqual(got, expected)
        del got
        self.memory_leak_teardown()

    @from_generic(pyfuncs_to_use)
    def check_only_shape(pyfunc, arr, shape, expected_shape):
        self.memory_leak_setup()
        got = generic_run(pyfunc, arr, shape)
        self.assertEqual(got.shape, expected_shape)
        self.assertEqual(got.size, arr.size)
        del got
        self.memory_leak_teardown()

    @from_generic(pyfuncs_to_use)
    def check_err_shape(pyfunc, arr, shape):
        with self.assertRaises(NotImplementedError) as raises:
            generic_run(pyfunc, arr, shape)
        self.assertEqual(str(raises.exception), 'incompatible shape for array')

    @from_generic(pyfuncs_to_use)
    def check_err_size(pyfunc, arr, shape):
        with self.assertRaises(ValueError) as raises:
            generic_run(pyfunc, arr, shape)
        self.assertEqual(str(raises.exception), 'total size of new array must be unchanged')

    @from_generic(pyfuncs_to_use)
    def check_err_multiple_negative(pyfunc, arr, shape):
        with self.assertRaises(ValueError) as raises:
            generic_run(pyfunc, arr, shape)
        self.assertEqual(str(raises.exception), 'multiple negative shape values')
    arr = np.arange(24)
    check(arr, (24,))
    check(arr, (4, 6))
    check(arr, (8, 3))
    check(arr, (8, 1, 3))
    check(arr, (1, 8, 1, 1, 3, 1))
    arr = np.arange(24).reshape((2, 3, 4))
    check(arr, (24,))
    check(arr, (4, 6))
    check(arr, (8, 3))
    check(arr, (8, 1, 3))
    check(arr, (1, 8, 1, 1, 3, 1))
    check_err_size(arr, ())
    check_err_size(arr, (25,))
    check_err_size(arr, (8, 4))
    arr = np.arange(24).reshape((1, 8, 1, 1, 3, 1))
    check(arr, (24,))
    check(arr, (4, 6))
    check(arr, (8, 3))
    check(arr, (8, 1, 3))
    arr = np.arange(24).reshape((2, 3, 4)).T
    check(arr, (4, 3, 2))
    check(arr, (1, 4, 1, 3, 1, 2, 1))
    check_err_shape(arr, (2, 3, 4))
    check_err_shape(arr, (6, 4))
    check_err_shape(arr, (2, 12))
    arr = np.arange(25).reshape(5, 5)
    check(arr, -1)
    check(arr, (-1,))
    check(arr, (-1, 5))
    check(arr, (5, -1, 5))
    check(arr, (5, 5, -1))
    check_err_size(arr, (-1, 4))
    check_err_multiple_negative(arr, (-1, -2, 5, 5))
    check_err_multiple_negative(arr, (5, 5, -1, -1))

    def check_empty(arr):
        check(arr, 0)
        check(arr, (0,))
        check(arr, (1, 0, 2))
        check(arr, (0, 55, 1, 0, 2))
        check_only_shape(arr, -1, (0,))
        check_only_shape(arr, (-1,), (0,))
        check_only_shape(arr, (0, -1), (0, 0))
        check_only_shape(arr, (4, -1), (4, 0))
        check_only_shape(arr, (-1, 0, 4), (0, 0, 4))
        check_err_size(arr, ())
        check_err_size(arr, 1)
        check_err_size(arr, (1, 2))
    arr = np.array([])
    check_empty(arr)
    check_empty(arr.reshape((3, 2, 0)))
    self.disable_leak_check()