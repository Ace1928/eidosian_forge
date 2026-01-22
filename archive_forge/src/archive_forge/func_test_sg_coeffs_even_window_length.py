import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_even_window_length():
    window_lengths = [4, 6, 8, 10, 12, 14, 16]
    for length in window_lengths:
        h_p_d = savgol_coeffs(length, 0, 0)
        assert_allclose(h_p_d, 1 / length)

    def h_p_d_closed_form_1(k, m):
        return 6 * (k - 0.5) / ((2 * m + 1) * m * (2 * m - 1))

    def h_p_d_closed_form_2(k, m):
        numer = 15 * (-4 * m ** 2 + 1 + 12 * (k - 0.5) ** 2)
        denom = 4 * (2 * m + 1) * (m + 1) * m * (m - 1) * (2 * m - 1)
        return numer / denom
    for length in window_lengths:
        m = length // 2
        expected_output = [h_p_d_closed_form_1(k, m) for k in range(-m + 1, m + 1)][::-1]
        actual_output = savgol_coeffs(length, 1, 1)
        assert_allclose(expected_output, actual_output)
        actual_output = savgol_coeffs(length, 2, 1)
        assert_allclose(expected_output, actual_output)
        expected_output = [h_p_d_closed_form_2(k, m) for k in range(-m + 1, m + 1)][::-1]
        actual_output = savgol_coeffs(length, 2, 2)
        assert_allclose(expected_output, actual_output)
        actual_output = savgol_coeffs(length, 3, 2)
        assert_allclose(expected_output, actual_output)