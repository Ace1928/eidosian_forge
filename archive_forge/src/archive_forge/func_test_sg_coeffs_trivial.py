import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_trivial():
    h = savgol_coeffs(1, 0)
    assert_allclose(h, [1])
    h = savgol_coeffs(3, 2)
    assert_allclose(h, [0, 1, 0], atol=1e-10)
    h = savgol_coeffs(5, 4)
    assert_allclose(h, [0, 0, 1, 0, 0], atol=1e-10)
    h = savgol_coeffs(5, 4, pos=1)
    assert_allclose(h, [0, 0, 0, 1, 0], atol=1e-10)
    h = savgol_coeffs(5, 4, pos=1, use='dot')
    assert_allclose(h, [0, 1, 0, 0, 0], atol=1e-10)