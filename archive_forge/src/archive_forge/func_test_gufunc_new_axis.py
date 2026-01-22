import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_gufunc_new_axis(self):
    gufunc = _get_matmulcore_gufunc(dtype=float64)
    X = np.random.randn(10, 3, 3)
    Y = np.random.randn(3, 3)
    gold = np.matmul(X, Y)
    res1 = gufunc(X, Y)
    np.testing.assert_allclose(gold, res1)
    res2 = gufunc(X, np.tile(Y, (10, 1, 1)))
    np.testing.assert_allclose(gold, res2)