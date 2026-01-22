import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_pickle_gufunc_dyanmic_null_init(self):
    """Dynamic gufunc w/o prepopulating before pickling.
        """

    @guvectorize('()->()', identity=1)
    def double(x, out):
        out[:] = x * 2
    ser = pickle.dumps(double)
    cloned = pickle.loads(ser)
    self.assertEqual(cloned._frozen, double._frozen)
    self.assertEqual(cloned.identity, double.identity)
    self.assertEqual(cloned.is_dynamic, double.is_dynamic)
    self.assertEqual(cloned.gufunc_builder._sigs, double.gufunc_builder._sigs)
    self.assertFalse(cloned._frozen)
    expect = np.zeros(1)
    got = np.zeros(1)
    double(0.5, out=expect)
    cloned(0.5, out=got)
    self.assertPreciseEqual(expect, got)
    arr = np.arange(10)
    expect = np.zeros_like(arr)
    got = np.zeros_like(arr)
    double(arr, out=expect)
    cloned(arr, out=got)
    self.assertPreciseEqual(expect, got)