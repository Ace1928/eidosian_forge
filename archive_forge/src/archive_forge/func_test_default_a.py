from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def test_default_a(self):
    x = arange(3.0, dtype=self.dtype)
    y = arange(3.0, dtype=x.dtype)
    real_y = x * 1.0 + y
    y = self.blas_func(x, y)
    assert_array_equal(real_y, y)