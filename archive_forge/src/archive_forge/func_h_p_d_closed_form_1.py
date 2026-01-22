import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def h_p_d_closed_form_1(k, m):
    return 6 * (k - 0.5) / ((2 * m + 1) * m * (2 * m - 1))