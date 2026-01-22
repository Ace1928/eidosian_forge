from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_change_view(self):
    pyfunc = numpy_broadcast_to
    cfunc = jit(nopython=True)(pyfunc)
    input_array = np.zeros(2, dtype=np.int32)
    shape = (2, 2)
    view = cfunc(input_array, shape)
    input_array[0] = 10
    self.assertEqual(input_array.sum(), 10)
    self.assertEqual(view.sum(), 20)