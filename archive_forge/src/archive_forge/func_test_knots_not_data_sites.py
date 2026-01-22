import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_knots_not_data_sites(self):
    k = 2
    t = np.r_[(self.xx[0],) * (k + 1), (self.xx[1:] + self.xx[:-1]) / 2.0, (self.xx[-1],) * (k + 1)]
    b = make_interp_spline(self.xx, self.yy, k, t, bc_type=([(2, 0)], [(2, 0)]))
    assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
    assert_allclose([b(self.xx[0], 2), b(self.xx[-1], 2)], [0.0, 0.0], atol=1e-14)