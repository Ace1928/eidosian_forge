from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_y_stride(self):
    alpha, beta, a, x, y = self.get_data(y_stride=2)
    desired_y = y.copy()
    desired_y[::2] = alpha * matrixmultiply(a, x) + beta * y[::2]
    y = self.blas_func(alpha, a, x, beta, y, incy=2)
    assert_array_almost_equal(desired_y, y)