import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_interp_edges():
    t = np.linspace(-5, 5, 21)
    delta = t[1] - t[0]
    x = np.array([t, 3 * t ** 2, t ** 3 - t])
    dx = np.array([np.ones_like(t), 6 * t, 3 * t ** 2 - 1.0])
    d2x = np.array([np.zeros_like(t), np.full_like(t, 6), 6 * t])
    window_length = 7
    y = savgol_filter(x, window_length, 3, axis=-1, mode='interp')
    assert_allclose(y, x, atol=1e-12)
    y1 = savgol_filter(x, window_length, 3, axis=-1, mode='interp', deriv=1, delta=delta)
    assert_allclose(y1, dx, atol=1e-12)
    y2 = savgol_filter(x, window_length, 3, axis=-1, mode='interp', deriv=2, delta=delta)
    assert_allclose(y2, d2x, atol=1e-12)
    x = x.T
    dx = dx.T
    d2x = d2x.T
    y = savgol_filter(x, window_length, 3, axis=0, mode='interp')
    assert_allclose(y, x, atol=1e-12)
    y1 = savgol_filter(x, window_length, 3, axis=0, mode='interp', deriv=1, delta=delta)
    assert_allclose(y1, dx, atol=1e-12)
    y2 = savgol_filter(x, window_length, 3, axis=0, mode='interp', deriv=2, delta=delta)
    assert_allclose(y2, d2x, atol=1e-12)