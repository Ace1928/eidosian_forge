import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_vectorize_only_kws(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import mul
    a = np.linspace(0, 1, 10)
    b = np.linspace(1, 2, 10)
    ufunc = vectorize(identity=PyUFunc_One, nopython=True)(mul)
    self.assertPreciseEqual(ufunc(a, b), a * b)