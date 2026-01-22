import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_npm_call_implicit_output(self):
    duadd = self.nopython_dufunc(pyuadd)

    @njit
    def npmadd(a0, a1):
        return duadd(a0, a1)
    X = np.linspace(0, 1.9, 20)
    X0 = X[:10]
    X1 = X[10:]
    out0 = npmadd(X0, X1)
    np.testing.assert_array_equal(X0 + X1, out0)
    Y0 = X0.reshape((2, 5))
    Y1 = X1.reshape((2, 5))
    out1 = npmadd(Y0, Y1)
    np.testing.assert_array_equal(Y0 + Y1, out1)
    Y2 = X1[:5]
    out2 = npmadd(Y0, Y2)
    np.testing.assert_array_equal(Y0 + Y2, out2)
    out3 = npmadd(1.0, 2.0)
    self.assertEqual(out3, 3.0)