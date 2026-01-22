import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_order_filter_asymmetric(self):
    x = np.arange(25).reshape(5, 5)
    domain = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 5, 6, 7, 8], [0, 10, 11, 12, 13], [0, 15, 16, 17, 18]])
    assert_allclose(order_filter(x, domain, 0), expected)
    expected = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
    assert_allclose(order_filter(x, domain, 1), expected)