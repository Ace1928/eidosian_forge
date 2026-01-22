import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_optional_float(self):

    def pyfunc(x, y):
        if y is None:
            return x
        else:
            return x + y
    cfunc = njit('(float64, optional(float64))')(pyfunc)
    self.assertAlmostEqual(pyfunc(1.0, 12.3), cfunc(1.0, 12.3))
    self.assertAlmostEqual(pyfunc(1.0, None), cfunc(1.0, None))