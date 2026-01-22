from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_y_stride_assert(self):
    alpha, beta, a, x, y = self.get_data(y_stride=2)
    with pytest.raises(Exception, match='failed for 2nd keyword'):
        y = self.blas_func(1, a, x, 1, y, trans=0, incy=3)
    with pytest.raises(Exception, match='failed for 2nd keyword'):
        y = self.blas_func(1, a, x, 1, y, trans=1, incy=3)