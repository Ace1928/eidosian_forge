import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_2d_repeated_axis(self):
    i, j = np.ogrid[:3, :4]
    arr = 10 * i + j
    arr_view = sliding_window_view(arr, (2, 3), (1, 1))
    expected = np.array([[[[0, 1, 2], [1, 2, 3]]], [[[10, 11, 12], [11, 12, 13]]], [[[20, 21, 22], [21, 22, 23]]]])
    assert_array_equal(arr_view, expected)