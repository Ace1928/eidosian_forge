import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_creation_from_Smooth(self):
    for nux, nuy in self.orders:
        lut_der = self.lut_smooth.partial_derivative(nux, nuy)
        a = lut_der(5.5, 5.5, grid=False)
        b = self.lut_smooth(5.5, 5.5, dx=nux, dy=nuy, grid=False)
        assert_equal(a, b)