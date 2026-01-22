import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polyint_basic(self):
    pyfunc = polyint
    cfunc = njit(polyint)
    self.assertPreciseEqual(pyfunc([1, 2, 3]), cfunc([1, 2, 3]))
    for i in range(2, 5):
        self.assertPreciseEqual(pyfunc([0], m=i), cfunc([0], m=i))
    for i in range(5):
        pol = [0] * i + [1]
        self.assertPreciseEqual(pyfunc(pol, m=1), pyfunc(pol, m=1))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            self.assertPreciseEqual(pyfunc(pol, m=j), cfunc(pol, m=j))
    c2 = np.array([[0, 1], [0, 2]])
    self.assertPreciseEqual(pyfunc(c2), cfunc(c2))
    c3 = np.arange(8).reshape((2, 2, 2))
    self.assertPreciseEqual(pyfunc(c3), cfunc(c3))