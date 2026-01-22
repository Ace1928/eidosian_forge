from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_sliding_window_view_errors(self):

    def _raises(msg, *args):
        with self.assertRaises(ValueError) as raises:
            sliding_window_view(*args)
        self.assertIn(msg, str(raises.exception))

    def _typing_error(msg, *args):
        with self.assertRaises(errors.TypingError) as raises:
            sliding_window_view(*args)
        self.assertIn(msg, str(raises.exception))
    self.disable_leak_check()
    arr1 = np.arange(24)
    arr2 = np.arange(200).reshape(10, 20)
    with self.subTest('1d window shape too large'):
        _raises('window_shape cannot be larger', arr1, 25, None)
    with self.subTest('2d window shape too large'):
        _raises('window_shape cannot be larger', arr2, (4, 21), None)
    with self.subTest('1d window negative size'):
        _raises('`window_shape` cannot contain negative', arr1, -1, None)
    with self.subTest('2d window with a negative size'):
        _raises('`window_shape` cannot contain negative', arr2, (4, -3), None)
    with self.subTest('1d array, 2d window shape'):
        _raises('matching length window_shape and axis', arr1, (10, 2), None)
    with self.subTest('2d window shape, only one axis given'):
        _raises('matching length window_shape and axis', arr2, (10, 2), 1)
    with self.subTest('1d window shape, 2 axes given'):
        _raises('matching length window_shape and axis', arr1, 5, (0, 0))
    with self.subTest('1d array, second axis'):
        _raises('Argument axis out of bounds', arr1, 4, 1)
    with self.subTest('1d array, axis -2'):
        _raises('Argument axis out of bounds', arr1, 4, -2)
    with self.subTest('2d array, fourth axis'):
        _raises('Argument axis out of bounds', arr2, (4, 4), (0, 3))
    with self.subTest('2d array, axis -3'):
        _raises('Argument axis out of bounds', arr2, (4, 4), (0, -3))
    with self.subTest('window_shape=None'):
        _typing_error('window_shape must be an integer or tuple of integer', arr1, None)
    with self.subTest('window_shape=float'):
        _typing_error('window_shape must be an integer or tuple of integer', arr1, 3.1)
    with self.subTest('window_shape=tuple(float)'):
        _typing_error('window_shape must be an integer or tuple of integer', arr1, (3.1,))
    with self.subTest('axis=float'):
        _typing_error('axis must be None, an integer or tuple of integer', arr1, 4, 3.1)
    with self.subTest('axis=tuple(float)'):
        _typing_error('axis must be None, an integer or tuple of integer', arr1, 4, (3.1,))