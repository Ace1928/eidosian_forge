import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def check_divmod_int(self, pyfunc, values):
    """
        Test 1 % 0 and 0 % 0.
        """
    f = vectorize(nopython=True)(pyfunc)
    a = np.array([5, 6, 0, 9])
    b = np.array([1, 0, 0, 4])
    expected = np.array(values)
    with self.check_warnings([]):
        res = f(a, b)
        self.assertPreciseEqual(res, expected)