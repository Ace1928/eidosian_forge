import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_trimseq_exception(self):
    cfunc = njit(trimseq)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc')
    self.assertIn('The argument "seq" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.arange(10).reshape(5, 2))
    self.assertIn('Coefficient array is not 1-d', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc((1, 2, 3, 0))
    self.assertIn('Unsupported type UniTuple(int64, 4) for argument "seq"', str(e.exception))