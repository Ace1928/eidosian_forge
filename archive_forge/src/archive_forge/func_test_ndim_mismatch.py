import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_ndim_mismatch(self):
    with self.assertRaises(TypeError) as raises:

        @guvectorize(['int32[:], int32[:]'], '(m,n)->(n)', target=self.target)
        def pyfunc(a, b):
            pass
    self.assertEqual('type and shape signature mismatch for arg #1', str(raises.exception))