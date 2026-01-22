import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_backward_compat(self):
    w1, gd1 = group_delay((1, 1))
    w2, gd2 = group_delay((1, 1), None)
    assert_array_almost_equal(w1, w2)
    assert_array_almost_equal(gd1, gd2)