import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_maxima(self):
    """h-maxima for various data types"""
    data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13], [13, 15, 40, 40, 18, 18, 18, 60, 60, 15], [14, 16, 40, 40, 19, 19, 19, 60, 60, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [15, 16, 18, 19, 19, 20, 19, 19, 18, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [14, 16, 80, 80, 19, 19, 19, 100, 100, 16], [13, 15, 80, 80, 18, 18, 18, 100, 100, 15], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]], dtype=np.uint8)
    expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
        data = data.astype(dtype)
        out = extrema.h_maxima(data, 40)
        error = diff(expected_result, out)
        assert error < eps