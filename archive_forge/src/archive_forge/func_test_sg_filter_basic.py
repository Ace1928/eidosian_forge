import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_basic():
    x = np.array([1.0, 2.0, 1.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_allclose(y, [1.0, 4.0 / 3, 1.0])
    y = savgol_filter(x, 3, 1, mode='mirror')
    assert_allclose(y, [5.0 / 3, 4.0 / 3, 5.0 / 3])
    y = savgol_filter(x, 3, 1, mode='wrap')
    assert_allclose(y, [4.0 / 3, 4.0 / 3, 4.0 / 3])