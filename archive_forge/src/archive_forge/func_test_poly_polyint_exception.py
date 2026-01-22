import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polyint_exception(self):
    cfunc = njit(polyint)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc')
    self.assertIn('The argument "c" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([1, 2, 3]), 'abc')
    self.assertIn('The argument "m" must be an integer', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(['a', 'b', 'c'], 1)
    self.assertIn('Input dtype must be scalar.', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(('a', 'b', 'c'), 1)
    self.assertIn('Input dtype must be scalar.', str(raises.exception))