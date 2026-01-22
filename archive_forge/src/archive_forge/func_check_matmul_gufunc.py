import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def check_matmul_gufunc(gufunc, A, B, C):
    Gold = np.matmul(A, B)
    gufunc(A, B, C)
    np.testing.assert_allclose(C, Gold, rtol=1e-05, atol=1e-08)