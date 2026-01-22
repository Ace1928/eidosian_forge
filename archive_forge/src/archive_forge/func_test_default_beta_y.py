from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_default_beta_y(self):
    alpha, beta, a, x, y = self.get_data()
    desired_y = matrixmultiply(a, x)
    y = self.blas_func(1, a, x)
    assert_array_almost_equal(desired_y, y)