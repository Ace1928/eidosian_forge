import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def compare_coeffs_to_alt(window_length, order):
    for pos in [None] + list(range(window_length)):
        h1 = savgol_coeffs(window_length, order, pos=pos, use='dot')
        h2 = alt_sg_coeffs(window_length, order, pos=pos)
        assert_allclose(h1, h2, atol=1e-10, err_msg='window_length = %d, order = %d, pos = %s' % (window_length, order, pos))