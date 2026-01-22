from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_shapes_raises(self):
    pyfunc = numpy_broadcast_shapes
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    data = [[(3,), (4,)], [(2, 3), (2,)], [(3,), (3,), (4,)], [(1, 3, 4), (2, 3, 3)], [(1, 2), (3, 1), (3, 2), (10, 5)], [2, (2, 3)]]
    for input_shape in data:
        with self.assertRaises(ValueError) as raises:
            cfunc(*input_shape)
        self.assertIn('shape mismatch: objects cannot be broadcast to a single shape', str(raises.exception))