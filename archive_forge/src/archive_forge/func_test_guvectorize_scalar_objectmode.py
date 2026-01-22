import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_guvectorize_scalar_objectmode(self):
    """
        Test passing of scalars to object mode gufuncs.
        """
    from numba.tests.npyufunc.ufuncbuilding_usecases import guadd_scalar_obj
    ufunc = guvectorize(['(int32[:,:], int32, int32[:,:])'], '(x,y),()->(x,y)', forceobj=True)(guadd_scalar_obj)
    a = np.arange(10, dtype='int32').reshape(2, 5)
    b = ufunc(a, 3)
    self.assertPreciseEqual(a + 3, b)