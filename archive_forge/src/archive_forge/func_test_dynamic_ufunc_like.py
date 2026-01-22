import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_dynamic_ufunc_like(self):

    def check_ufunc_output(gufunc, x):
        out = np.zeros(10, dtype=x.dtype)
        out_kw = np.zeros(10, dtype=x.dtype)
        gufunc(x, x, x, out)
        gufunc(x, x, x, out=out_kw)
        golden = x * x + x
        np.testing.assert_equal(out, golden)
        np.testing.assert_equal(out_kw, golden)
    gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target, is_dynamic=True)
    x = np.arange(10, dtype=np.intp)
    check_ufunc_output(gufunc, x)