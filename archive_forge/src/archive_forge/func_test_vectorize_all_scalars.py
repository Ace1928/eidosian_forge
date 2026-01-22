import numpy as np
from numba import vectorize
from numba import cuda, float64
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_vectorize_all_scalars(self):

    @vectorize(sig, target='cuda')
    def vector_add(a, b):
        return a + b
    v = vector_add(1.0, 1.0)
    np.testing.assert_almost_equal(2.0, v)