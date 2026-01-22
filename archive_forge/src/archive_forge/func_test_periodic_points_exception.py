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
def test_periodic_points_exception(self):
    np.random.seed(1234)
    k = 5
    n = 8
    x = np.sort(np.random.random_sample(n))
    y = np.random.random_sample(n)
    y[0] = y[-1] - 1
    with assert_raises(ValueError):
        make_interp_spline(x, y, k=k, bc_type='periodic')