import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_Polynomial_exeption(self):

    def pyfunc3(c, dom, win):
        p = poly.Polynomial(c, dom, win)
        return p
    cfunc3 = njit(pyfunc3)
    self.disable_leak_check()
    input2 = np.array([1, 2])
    input3 = np.array([1, 2, 3])
    input2D = np.arange(4).reshape((2, 2))
    with self.assertRaises(ValueError) as raises:
        cfunc3(input2, input3, input2)
    self.assertIn('Domain has wrong number of elements.', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc3(input2, input2, input3)
    self.assertIn('Window has wrong number of elements.', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc3(input2D, input2, input2)
    self.assertIn('Coefficient array is not 1-d', str(raises.exception))