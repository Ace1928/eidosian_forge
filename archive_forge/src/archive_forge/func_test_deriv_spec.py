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
def test_deriv_spec(self):
    x = y = [1.0, 2, 3, 4, 5, 6]
    with assert_raises(ValueError):
        make_interp_spline(x, y, bc_type=([(1, 0.0)], None))
    with assert_raises(ValueError):
        make_interp_spline(x, y, bc_type=(1, 0.0))
    with assert_raises(ValueError):
        make_interp_spline(x, y, bc_type=[(1, 0.0)])
    with assert_raises(ValueError):
        make_interp_spline(x, y, bc_type=42)
    l, r = ((1, 0.0), (1, 0.0))
    with assert_raises(ValueError):
        make_interp_spline(x, y, bc_type=(l, r))