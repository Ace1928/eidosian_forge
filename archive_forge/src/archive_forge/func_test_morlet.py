import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_morlet(self):
    with pytest.deprecated_call():
        x = wavelets.morlet(50, 4.1, complete=True)
        y = wavelets.morlet(50, 4.1, complete=False)
        assert_equal(len(x), len(y))
        assert_array_less(x, y)
        x = wavelets.morlet(10, 50, complete=False)
        y = wavelets.morlet(10, 50, complete=True)
        assert_equal(x, y)
        x = np.array([1.73752399e-09 + 9.84327394e-25j, 0.649471756 + 0j, 1.73752399e-09 - 9.84327394e-25j])
        y = wavelets.morlet(3, w=2, complete=True)
        assert_array_almost_equal(x, y)
        x = np.array([2.00947715e-09 + 9.84327394e-25j, 0.751125544 + 0j, 2.00947715e-09 - 9.84327394e-25j])
        y = wavelets.morlet(3, w=2, complete=False)
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, s=4, complete=True)
        y = wavelets.morlet(20000, s=8, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, s=4, complete=False)
        assert_array_almost_equal(y, x, decimal=2)
        y = wavelets.morlet(20000, s=8, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, w=3, s=5, complete=True)
        y = wavelets.morlet(20000, w=3, s=10, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, w=3, s=5, complete=False)
        assert_array_almost_equal(y, x, decimal=2)
        y = wavelets.morlet(20000, w=3, s=10, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, w=7, s=10, complete=True)
        y = wavelets.morlet(20000, w=7, s=20, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)
        x = wavelets.morlet(10000, w=7, s=10, complete=False)
        assert_array_almost_equal(x, y, decimal=2)
        y = wavelets.morlet(20000, w=7, s=20, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)