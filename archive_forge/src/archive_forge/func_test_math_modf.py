import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_modf(self):

    def modf_template_nan(dtype, arytype):
        A = np.array([np.nan], dtype=dtype)
        B = np.zeros_like(A)
        C = np.zeros_like(A)
        cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
        cfunc[1, len(A)](A, B, C)
        self.assertTrue(np.isnan(B))
        self.assertTrue(np.isnan(C))

    def modf_template_compare(A, dtype, arytype):
        A = A.astype(dtype)
        B = np.zeros_like(A)
        C = np.zeros_like(A)
        cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
        cfunc[1, len(A)](A, B, C)
        D, E = np.modf(A)
        self.assertTrue(np.array_equal(B, D))
        self.assertTrue(np.array_equal(C, E))
    nelem = 50
    with self.subTest('float32 modf on simple float'):
        modf_template_compare(np.linspace(0, 10, nelem), dtype=np.float32, arytype=float32[:])
    with self.subTest('float32 modf on +- infinity'):
        modf_template_compare(np.array([np.inf, -np.inf]), dtype=np.float32, arytype=float32[:])
    with self.subTest('float32 modf on nan'):
        modf_template_nan(dtype=np.float32, arytype=float32[:])
    with self.subTest('float64 modf on simple float'):
        modf_template_compare(np.linspace(0, 10, nelem), dtype=np.float64, arytype=float64[:])
    with self.subTest('float64 modf on +- infinity'):
        modf_template_compare(np.array([np.inf, -np.inf]), dtype=np.float64, arytype=float64[:])
    with self.subTest('float64 modf on nan'):
        modf_template_nan(dtype=np.float64, arytype=float64[:])