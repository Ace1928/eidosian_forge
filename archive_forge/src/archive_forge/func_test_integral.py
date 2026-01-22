import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_integral(self):
    x = [1, 1, 1, 2, 2, 2, 4, 4, 4]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    z = array([0, 7, 8, 3, 4, 7, 1, 3, 4])
    with suppress_warnings() as sup:
        sup.filter(UserWarning, '\nThe required storage space')
        lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1, s=0)
    tx = [1, 2, 4]
    ty = [1, 2, 3]
    tz = lut(tx, ty)
    trpz = 0.25 * (diff(tx)[:, None] * diff(ty)[None, :] * (tz[:-1, :-1] + tz[1:, :-1] + tz[:-1, 1:] + tz[1:, 1:])).sum()
    assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz)
    lut2 = SmoothBivariateSpline(x, y, z, kx=2, ky=2, s=0)
    assert_almost_equal(lut2.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz, decimal=0)
    tz = lut(tx[:-1], ty[:-1])
    trpz = 0.25 * (diff(tx[:-1])[:, None] * diff(ty[:-1])[None, :] * (tz[:-1, :-1] + tz[1:, :-1] + tz[:-1, 1:] + tz[1:, 1:])).sum()
    assert_almost_equal(lut.integral(tx[0], tx[-2], ty[0], ty[-2]), trpz)