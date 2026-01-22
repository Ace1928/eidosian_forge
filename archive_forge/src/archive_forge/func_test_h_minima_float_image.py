import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_minima_float_image(self):
    """specific tests for h-minima float image type"""
    w = 10
    x, y = np.mgrid[0:w, 0:w]
    data = 180 + 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
    data[2:4, 2:4] = 160
    data[2:4, 7:9] = 140
    data[7:9, 2:4] = 120
    data[7:9, 7:9] = 100
    data = data.astype(np.float32)
    expected_result = np.zeros_like(data)
    expected_result[data < 180.1] = 1.0
    for h in [1e-12, 1e-06, 0.001, 0.01, 0.1, 0.1]:
        out = extrema.h_minima(data, h)
        error = diff(expected_result, out)
        assert error < eps