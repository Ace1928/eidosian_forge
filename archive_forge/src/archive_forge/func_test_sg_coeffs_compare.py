import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_compare():
    for window_length in range(1, 8, 2):
        for order in range(window_length):
            compare_coeffs_to_alt(window_length, order)