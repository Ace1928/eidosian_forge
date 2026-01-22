import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def check_polyder(p, m, expected):
    dp = _polyder(p, m)
    assert_array_equal(dp, expected)