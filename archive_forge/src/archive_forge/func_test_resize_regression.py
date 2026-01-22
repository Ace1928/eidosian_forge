import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_resize_regression(self):
    """Regression test for #1375."""
    x = [-1.0, -0.65016502, -0.58856235, -0.26903553, -0.17370892, -0.10011001, 0.0, 0.10011001, 0.17370892, 0.26903553, 0.58856235, 0.65016502, 1.0]
    y = [1.0, 0.62928599, 0.5797223, 0.39965815, 0.36322694, 0.3508061, 0.35214793, 0.3508061, 0.36322694, 0.39965815, 0.5797223, 0.62928599, 1.0]
    w = [1000000000000.0, 688.875973, 489.314737, 426.864807, 607.74677, 451.341444, 317.48021, 451.341444, 607.74677, 426.864807, 489.314737, 688.875973, 1000000000000.0]
    spl = UnivariateSpline(x=x, y=y, w=w, s=None)
    desired = array([0.35100374, 0.51715855, 0.87789547, 0.98719344])
    assert_allclose(spl([0.1, 0.5, 0.9, 0.99]), desired, atol=0.0005)