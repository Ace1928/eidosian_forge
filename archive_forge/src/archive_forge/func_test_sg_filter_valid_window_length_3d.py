import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_valid_window_length_3d():
    """Tests that the window_length check is using the correct axis."""
    x = np.ones((10, 20, 30))
    savgol_filter(x, window_length=29, polyorder=3, mode='interp')
    with pytest.raises(ValueError, match='window_length must be less than'):
        savgol_filter(x, window_length=31, polyorder=3, mode='interp')
    savgol_filter(x, window_length=9, polyorder=3, axis=0, mode='interp')
    with pytest.raises(ValueError, match='window_length must be less than'):
        savgol_filter(x, window_length=11, polyorder=3, axis=0, mode='interp')