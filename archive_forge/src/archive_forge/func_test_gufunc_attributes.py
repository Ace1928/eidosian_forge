import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_gufunc_attributes(self):

    @guvectorize('(n)->(n)')
    def gufunc(x, res):
        acc = 0
        for i in range(x.shape[0]):
            acc += x[i]
            res[i] = acc
    attrs = ['signature', 'accumulate', 'at', 'outer', 'reduce', 'reduceat']
    for attr in attrs:
        contains = hasattr(gufunc, attr)
        self.assertTrue(contains, 'dynamic gufunc not exporting "%s"' % (attr,))
    a = np.array([1, 2, 3, 4])
    res = np.array([0, 0, 0, 0])
    gufunc(a, res)
    self.assertPreciseEqual(res, np.array([1, 3, 6, 10]))
    self.assertEqual(gufunc.signature, '(n)->(n)')
    with self.assertRaises(RuntimeError) as raises:
        gufunc.accumulate(a)
    self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
    with self.assertRaises(RuntimeError) as raises:
        gufunc.reduce(a)
    self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
    with self.assertRaises(RuntimeError) as raises:
        gufunc.reduceat(a, [0, 2])
    self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
    with self.assertRaises(TypeError) as raises:
        gufunc.outer(a, a)
    self.assertEqual(str(raises.exception), 'method outer is not allowed in ufunc with non-trivial signature')