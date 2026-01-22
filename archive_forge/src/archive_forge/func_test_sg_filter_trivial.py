import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_trivial():
    """ Test some trivial edge cases for savgol_filter()."""
    x = np.array([1.0])
    y = savgol_filter(x, 1, 0)
    assert_equal(y, [1.0])
    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_almost_equal(y, [1.0], decimal=15)
    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='nearest')
    assert_almost_equal(y, [3.0], decimal=15)
    x = np.array([1.0] * 3)
    y = savgol_filter(x, 3, 1, mode='wrap')
    assert_almost_equal(y, [1.0, 1.0, 1.0], decimal=15)