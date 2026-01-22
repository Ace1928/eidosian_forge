import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_gufunc_struct(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import guadd
    gufb = GUFuncBuilder(guadd, '(x, y),(x, y)->(x, y)')
    cres = gufb.add('void(complex64[:,:], complex64[:,:], complex64[:,:])')
    self.assertFalse(cres.objectmode)
    ufunc = gufb.build_ufunc()
    a = np.arange(10, dtype='complex64').reshape(2, 5) + 1j
    b = ufunc(a, a)
    self.assertPreciseEqual(a + a, b)