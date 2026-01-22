import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_integral_out_of_bounds(self):
    x = np.linspace(0.0, 1.0, 7)
    for ext in range(4):
        f = UnivariateSpline(x, x, s=0, ext=ext)
        for a, b in [(1, 1), (1, 5), (2, 5), (0, 0), (-2, 0), (-2, -1)]:
            assert_allclose(f.integral(a, b), 0, atol=1e-15)