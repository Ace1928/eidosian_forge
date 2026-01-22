import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_medfilt_order_filter(self):
    x = np.arange(25).reshape(5, 5)
    expected = np.array([[0, 1, 2, 3, 0], [1, 6, 7, 8, 4], [6, 11, 12, 13, 9], [11, 16, 17, 18, 14], [0, 16, 17, 18, 0]])
    assert_allclose(medfilt(x, 3), expected)
    assert_allclose(order_filter(x, np.ones((3, 3)), 4), expected)