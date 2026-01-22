import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_gufunc_old_style_scalar_as_array(self):

    @guvectorize(['void(int32[:],int32[:],int32[:])'], '(n),()->(n)', target='cuda')
    def gufunc(x, y, res):
        for i in range(x.shape[0]):
            res[i] = x[i] + y[0]
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([2], dtype=np.int32)
    res = np.zeros(4, dtype=np.int32)
    expected = res.copy()
    expected = a + b
    gufunc(a, b, out=res)
    np.testing.assert_almost_equal(expected, res)
    a = np.array([1, 2, 3, 4] * 2, dtype=np.int32).reshape(2, 4)
    b = np.array([2, 10], dtype=np.int32)
    res = np.zeros((2, 4), dtype=np.int32)
    expected = res.copy()
    expected[0] = a[0] + b[0]
    expected[1] = a[1] + b[1]
    gufunc(a, b, res)
    np.testing.assert_almost_equal(expected, res)