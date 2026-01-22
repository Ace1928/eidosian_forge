import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_dynamic_matmul(self):

    def check_matmul_gufunc(gufunc, A, B, C):
        Gold = np.matmul(A, B)
        gufunc(A, B, C)
        np.testing.assert_allclose(C, Gold, rtol=1e-05, atol=1e-08)
    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target=self.target, is_dynamic=True)
    matrix_ct = 10
    Ai64 = np.arange(matrix_ct * 2 * 4, dtype=np.int64).reshape(matrix_ct, 2, 4)
    Bi64 = np.arange(matrix_ct * 4 * 5, dtype=np.int64).reshape(matrix_ct, 4, 5)
    Ci64 = np.arange(matrix_ct * 2 * 5, dtype=np.int64).reshape(matrix_ct, 2, 5)
    check_matmul_gufunc(gufunc, Ai64, Bi64, Ci64)
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
    C = np.arange(matrix_ct * 2 * 5, dtype=np.float32).reshape(matrix_ct, 2, 5)
    check_matmul_gufunc(gufunc, A, B, C)
    self.assertEqual(len(gufunc.types), 2)