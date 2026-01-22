import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_bessel_poly(self):
    assert_array_equal(_bessel_poly(5), [945, 945, 420, 105, 15, 1])
    assert_array_equal(_bessel_poly(4, True), [1, 10, 45, 105, 105])