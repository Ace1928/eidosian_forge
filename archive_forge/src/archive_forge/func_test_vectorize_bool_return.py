import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_vectorize_bool_return(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import equals
    ufunc = vectorize(['bool_(int32, int32)'])(equals)
    a = np.arange(10, dtype='int32')
    r = ufunc(a, a)
    self.assertPreciseEqual(r, np.ones(r.shape, dtype=np.bool_))