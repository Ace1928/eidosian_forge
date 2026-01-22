from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_non_array_input(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    outarrays = cfunc(np.intp(2), np.zeros((1, 3), dtype=np.intp))
    expected = [(1, 3), (1, 3)]
    got = [a.shape for a in outarrays]
    self.assertPreciseEqual(expected, got)