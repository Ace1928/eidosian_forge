import numpy as np
from numba import cuda
from numba.cuda.kernels.transpose import transpose
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
def test_transpose_view(self):
    a = np.arange(120, dtype=np.int64).reshape((10, 12))
    a_view_t = a[::2, ::2].T
    d_a = cuda.to_device(a)
    d_a_view_t = d_a[::2, ::2].T
    self.assertEqual(d_a_view_t.shape, (6, 5))
    self.assertEqual(d_a_view_t.strides, (40, 8))
    h_a_view_t = d_a_view_t.copy_to_host()
    np.testing.assert_array_equal(a_view_t, h_a_view_t)