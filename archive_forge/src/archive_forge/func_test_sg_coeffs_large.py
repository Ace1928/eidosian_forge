import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_large():
    coeffs0 = savgol_coeffs(31, 9)
    assert_array_almost_equal(coeffs0, coeffs0[::-1])
    coeffs1 = savgol_coeffs(31, 9, deriv=1)
    assert_array_almost_equal(coeffs1, -coeffs1[::-1])