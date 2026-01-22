import numpy as np
from numba import int8, int16, int32
from numba import cuda, vectorize, njit
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.enum_usecases import (
def test_int_coerce(self):

    def f(x, out):
        if x > RequestError.internal_error:
            out[0] = x - RequestError.not_found
        else:
            out[0] = x + Shape.circle
    cuda_f = cuda.jit(f)
    for x in [300, 450, 550]:
        got = np.zeros((1,), dtype=np.int32)
        expected = got.copy()
        cuda_f[1, 1](x, got)
        f(x, expected)
        self.assertPreciseEqual(expected, got)