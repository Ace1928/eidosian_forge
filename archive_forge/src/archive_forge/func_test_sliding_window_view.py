from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_sliding_window_view(self):

    def check(arr, window_shape, axis):
        expected = np.lib.stride_tricks.sliding_window_view(arr, window_shape, axis, writeable=True)
        got = sliding_window_view(arr, window_shape, axis)
        self.assertPreciseEqual(got, expected)
    arr1 = np.arange(24)
    for axis in [None, 0, -1, (0,)]:
        with self.subTest(f'1d array, axis={axis}'):
            check(arr1, 5, axis)
    arr2 = np.arange(200).reshape(10, 20)
    for axis in [0, -1]:
        with self.subTest(f'2d array, axis={axis}'):
            check(arr2, 5, axis)
    for axis in [None, (0, 1), (1, 0), (1, -2)]:
        with self.subTest(f'2d array, axis={axis}'):
            check(arr2, (5, 8), axis)
    arr4 = np.arange(200).reshape(4, 5, 5, 2)
    for axis in [(1, 2), (-2, -3)]:
        with self.subTest(f'4d array, axis={axis}'):
            check(arr4, (3, 2), axis)
    with self.subTest('2d array, repeated axes'):
        check(arr2, (5, 3, 3), (0, 1, 0))