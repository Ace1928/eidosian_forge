import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
from numba.tests.support import skip_unless_cffi
def test_ex_linking_cu(self):
    from numba import cuda
    import numpy as np
    import os
    mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')
    basedir = os.path.dirname(os.path.abspath(__file__))
    functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')

    @cuda.jit(link=[functions_cu])
    def multiply_vectors(r, x, y):
        i = cuda.grid(1)
        if i < len(r):
            r[i] = mul(x[i], y[i])
    N = 32
    np.random.seed(1)
    x = np.random.rand(N).astype(np.float32)
    y = np.random.rand(N).astype(np.float32)
    r = np.zeros_like(x)
    multiply_vectors[1, 32](r, x, y)
    np.testing.assert_array_equal(r, x * y)