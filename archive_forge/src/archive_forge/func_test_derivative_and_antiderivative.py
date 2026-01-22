import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_derivative_and_antiderivative(self):
    x = np.linspace(0, 1, 70) ** 3
    y = np.cos(x)
    spl = UnivariateSpline(x, y, s=0)
    spl2 = spl.antiderivative(2).derivative(2)
    assert_allclose(spl(0.3), spl2(0.3))
    spl2 = spl.antiderivative(1)
    assert_allclose(spl2(0.6) - spl2(0.2), spl.integral(0.2, 0.6))