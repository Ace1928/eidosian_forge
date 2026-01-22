import numpy
from numpy import fft
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
import pytest
from scipy import ndimage
@pytest.mark.parametrize('shape', [(0,), (0, 10), (10, 0)])
@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@pytest.mark.parametrize('test_func', [ndimage.fourier_ellipsoid, ndimage.fourier_gaussian, ndimage.fourier_uniform])
def test_fourier_zero_length_dims(self, shape, dtype, test_func):
    a = numpy.ones(shape, dtype)
    b = test_func(a, 3)
    assert_equal(a, b)