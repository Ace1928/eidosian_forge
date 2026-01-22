import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_pickle_gufunc_non_dyanmic(self):
    """Non-dynamic gufunc.
        """

    @guvectorize(['f8,f8[:]'], '()->()')
    def double(x, out):
        out[:] = x * 2
    ser = pickle.dumps(double)
    cloned = pickle.loads(ser)
    self.assertEqual(cloned._frozen, double._frozen)
    self.assertEqual(cloned.identity, double.identity)
    self.assertEqual(cloned.is_dynamic, double.is_dynamic)
    self.assertEqual(cloned.gufunc_builder._sigs, double.gufunc_builder._sigs)
    self.assertTrue(cloned._frozen)
    cloned.disable_compile()
    self.assertTrue(cloned._frozen)
    self.assertPreciseEqual(double(0.5), cloned(0.5))
    arr = np.arange(10)
    self.assertPreciseEqual(double(arr), cloned(arr))