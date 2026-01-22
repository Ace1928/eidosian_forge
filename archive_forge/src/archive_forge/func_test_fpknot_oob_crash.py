import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_fpknot_oob_crash(self):
    x = range(109)
    y = [0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 11.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 10.6, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 10.7, 0.0, 0.0, 10.5, 0.0, 0.0, 11.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 10.7, 0.0, 0.0, 10.9, 0.0, 0.0, 10.8, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 10.6, 0.0, 0.0, 0.0, 10.4, 0.0, 0.0, 10.6, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 10.4, 0.0, 0.0, 0.0, 10.8, 0.0]
    with suppress_warnings() as sup:
        r = sup.record(UserWarning, '\nThe maximal number of iterations maxit \\(set to 20 by the program\\)\nallowed for finding a smoothing spline with fp=s has been reached: s\ntoo small.\nThere is an approximation returned but the corresponding weighted sum\nof squared residuals does not satisfy the condition abs\\(fp-s\\)/s < tol.')
        UnivariateSpline(x, y, k=1)
        assert_equal(len(r), 1)