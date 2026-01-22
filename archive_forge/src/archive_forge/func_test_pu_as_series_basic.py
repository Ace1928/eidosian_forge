import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_pu_as_series_basic(self):
    pyfunc1 = polyasseries1
    cfunc1 = njit(polyasseries1)
    pyfunc2 = polyasseries2
    cfunc2 = njit(polyasseries2)

    def inputs():
        yield np.arange(4)
        yield np.arange(6).reshape((2, 3))
        yield (1, np.arange(3), np.arange(2, dtype=np.float32))
        yield ([1, 2, 3, 4, 0], [1, 2, 3])
        yield ((0, 0, 0.001, 0, 1e-05, 0, 0), (1, 2, 3, 4, 5, 6, 7))
        yield ((0, 0, 0.001, 0, 1e-05, 0, 0), (1j, 2, 3j, 4j, 5, 6j, 7))
        yield (2, [1.1, 0.0])
        yield ([1, 2, 3, 0],)
        yield ((1, 2, 3, 0),)
        yield (np.array([1, 2, 3, 0]),)
        yield [np.array([1, 2, 3, 0]), np.array([1, 2, 3, 0])]
        yield [np.array([1, 2, 3])]
    for input in inputs():
        self.assertPreciseEqual(pyfunc1(input), cfunc1(input))
        self.assertPreciseEqual(pyfunc2(input, False), cfunc2(input, False))
        self.assertPreciseEqual(pyfunc2(input, True), cfunc2(input, True))