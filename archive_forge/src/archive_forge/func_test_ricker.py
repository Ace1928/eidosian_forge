import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_ricker(self):
    with pytest.deprecated_call():
        w = wavelets.ricker(1.0, 1)
        expected = 2 / (np.sqrt(3 * 1.0) * np.pi ** 0.25)
        assert_array_equal(w, expected)
        lengths = [5, 11, 15, 51, 101]
        for length in lengths:
            w = wavelets.ricker(length, 1.0)
            assert_(len(w) == length)
            max_loc = np.argmax(w)
            assert_(max_loc == length // 2)
        points = 100
        w = wavelets.ricker(points, 2.0)
        half_vec = np.arange(0, points // 2)
        assert_array_almost_equal(w[half_vec], w[-(half_vec + 1)])
        aas = [5, 10, 15, 20, 30]
        points = 99
        for a in aas:
            w = wavelets.ricker(points, a)
            vec = np.arange(0, points) - (points - 1.0) / 2
            exp_zero1 = np.argmin(np.abs(vec - a))
            exp_zero2 = np.argmin(np.abs(vec + a))
            assert_array_almost_equal(w[exp_zero1], 0)
            assert_array_almost_equal(w[exp_zero2], 0)