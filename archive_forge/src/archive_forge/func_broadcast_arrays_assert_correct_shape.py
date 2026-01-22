from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def broadcast_arrays_assert_correct_shape(self, input_shapes, expected_shape):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    inarrays = [np.zeros(s) for s in input_shapes]
    outarrays = cfunc(*inarrays)
    expected = [expected_shape] * len(inarrays)
    got = [a.shape for a in outarrays]
    self.assertPreciseEqual(expected, got)