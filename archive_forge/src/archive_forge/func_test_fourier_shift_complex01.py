import numpy
from numpy import fft
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
import pytest
from scipy import ndimage
@pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
@pytest.mark.parametrize('dtype, dec', [(numpy.complex64, 6), (numpy.complex128, 11)])
def test_fourier_shift_complex01(self, shape, dtype, dec):
    expected = numpy.arange(shape[0] * shape[1], dtype=dtype)
    expected.shape = shape
    a = fft.fft(expected, shape[0], 0)
    a = fft.fft(a, shape[1], 1)
    a = ndimage.fourier_shift(a, [1, 1], -1, 0)
    a = fft.ifft(a, shape[1], 1)
    a = fft.ifft(a, shape[0], 0)
    assert_array_almost_equal(a.real[1:, 1:], expected[:-1, :-1], decimal=dec)
    assert_array_almost_equal(a.imag, numpy.zeros(shape), decimal=dec)