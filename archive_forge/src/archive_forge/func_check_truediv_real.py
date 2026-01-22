import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def check_truediv_real(self, dtype):
    """
        Test 1 / 0 and 0 / 0.
        """
    f = vectorize(nopython=True)(truediv)
    a = np.array([5.0, 6.0, 0.0, 8.0], dtype=dtype)
    b = np.array([1.0, 0.0, 0.0, 4.0], dtype=dtype)
    expected = np.array([5.0, float('inf'), float('nan'), 2.0])
    with self.check_warnings(['divide by zero encountered', 'invalid value encountered']):
        res = f(a, b)
        self.assertPreciseEqual(res, expected)