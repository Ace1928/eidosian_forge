from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_x_stride_assert(self):
    alpha, beta, a, x, y = self.get_data(x_stride=2)
    with pytest.raises(Exception, match='failed for 3rd argument'):
        y = self.blas_func(1, a, x, 1, y, trans=0, incx=3)
    with pytest.raises(Exception, match='failed for 3rd argument'):
        y = self.blas_func(1, a, x, 1, y, trans=1, incx=3)