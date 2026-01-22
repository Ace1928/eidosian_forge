import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_optional_array_error(self):

    def pyfunc(y):
        return y[0]
    cfunc = njit('(optional(int32[:]),)')(pyfunc)
    with self.assertRaises(TypeError) as raised:
        cfunc(None)
    self.assertIn('expected array(int32, 1d, A), got None', str(raised.exception))
    y = np.array([43981], dtype=np.int32)
    self.assertEqual(cfunc(y), pyfunc(y))