import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_strictly_increasing_x(self):
    xx = np.arange(10, dtype=float)
    yy = xx ** 3
    x = np.arange(10, dtype=float)
    x[1] = x[0]
    y = x ** 3
    w = np.ones_like(x)
    spl = UnivariateSpline(xx, yy, check_finite=True)
    t = spl.get_knots()[3:4]
    UnivariateSpline(x=x, y=y, w=w, s=1, check_finite=True)
    LSQUnivariateSpline(x=x, y=y, t=t, w=w, check_finite=True)
    assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, s=0, check_finite=True))
    assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, check_finite=True))