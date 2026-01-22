import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_derivatives_2(self):
    x = np.arange(8)
    y = x ** 3 + 2.0 * x ** 2
    tck = splrep(x, y, s=0)
    ders = spalde(3, tck)
    assert_allclose(ders, [45.0, 39.0, 22.0, 6.0], atol=1e-15)
    spl = UnivariateSpline(x, y, s=0, k=3)
    assert_allclose(spl.derivatives(3), ders, atol=1e-15)