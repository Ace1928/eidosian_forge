from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to(self):
    pyfunc = numpy_broadcast_to
    cfunc = jit(nopython=True)(pyfunc)
    data = [[np.array(0), (0,)], [np.array(0), (1,)], [np.array(0), (3,)], [np.ones(1), (1,)], [np.ones(1), (2,)], [np.ones(1), (1, 2, 3)], [np.arange(3), (3,)], [np.arange(3), (1, 3)], [np.arange(3), (2, 3)], [np.ones(0), 0], [np.ones(1), 1], [np.ones(1), 2], [np.ones(1), (0,)], [np.ones((1, 2)), (0, 2)], [np.ones((2, 1)), (2, 0)], [2, (2, 2)], [(1, 2), (2, 2)]]
    for input_array, shape in data:
        expected = pyfunc(input_array, shape)
        got = cfunc(input_array, shape)
        self.assertPreciseEqual(got, expected)