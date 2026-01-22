import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_nan(self):
    x = np.arange(10, dtype=float)
    y = x ** 3
    w = np.ones_like(x)
    spl = UnivariateSpline(x, y, check_finite=True)
    t = spl.get_knots()[3:4]
    y_end = y[-1]
    for z in [np.nan, np.inf, -np.inf]:
        y[-1] = z
        assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, LSQUnivariateSpline, **dict(x=x, y=y, t=t, check_finite=True))
        y[-1] = y_end
        w[-1] = z
        assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, w=w, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, w=w, check_finite=True))
        assert_raises(ValueError, LSQUnivariateSpline, **dict(x=x, y=y, t=t, w=w, check_finite=True))