import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_bspline(self):
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        assert_allclose(bsp.bspline([-1, 0, 1], 0), array([0, 1, 0]))
        assert_allclose(bsp.bspline([-1, 0, 1], 1), array([0, 1, 0]))
        assert_allclose(bsp.bspline([-2, -1, 0, 1, 2], 2), array([0, 1, 6, 1, 0]) / 8.0)
        assert_allclose(bsp.bspline([-2, -1, 0, 1, 2], 3), array([0, 1, 4, 1, 0]) / 6.0)
        assert_allclose(bsp.bspline([-3, -2, -1, 0, 1, 2, 3], 4), array([0, 1, 76, 230, 76, 1, 0]) / 384.0)
        assert_allclose(bsp.bspline([-3, -2, -1, 0, 1, 2, 3], 5), array([0, 1, 26, 66, 26, 1, 0]) / 120.0)
        np.random.seed(12458)
        assert_allclose(bsp.bspline(np.random.rand(1, 1), 2), array([[0.73694695]]))