from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_two_compatible_by_ones_input_shapes(self):
    data = [[[(1,), (3,)], (3,)], [[(1, 3), (3, 3)], (3, 3)], [[(3, 1), (3, 3)], (3, 3)], [[(1, 3), (3, 1)], (3, 3)], [[(1, 1), (3, 3)], (3, 3)], [[(1, 1), (1, 3)], (1, 3)], [[(1, 1), (3, 1)], (3, 1)], [[(1, 0), (0, 0)], (0, 0)], [[(0, 1), (0, 0)], (0, 0)], [[(1, 0), (0, 1)], (0, 0)], [[(1, 1), (0, 0)], (0, 0)], [[(1, 1), (1, 0)], (1, 0)], [[(1, 1), (0, 1)], (0, 1)]]
    for input_shapes, expected_shape in data:
        self.broadcast_arrays_assert_correct_shape(input_shapes, expected_shape)
        self.broadcast_arrays_assert_correct_shape(input_shapes[::-1], expected_shape)