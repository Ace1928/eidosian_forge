from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_0d_array(self):
    pyfunc = numpy_broadcast_to
    cfunc = jit(nopython=True)(pyfunc)
    inputs = [np.array(123), 123, True]
    shape = ()
    for arr in inputs:
        expected = pyfunc(arr, shape)
        got = cfunc(arr, shape)
        self.assertPreciseEqual(expected, got)
        self.assertFalse(got.flags['WRITEABLE'])