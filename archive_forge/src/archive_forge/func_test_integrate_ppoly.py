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
def test_integrate_ppoly(self):
    x = [0, 1, 2, 3, 4]
    b = make_interp_spline(x, x)
    b.extrapolate = 'periodic'
    p = PPoly.from_spline(b)
    for x0, x1 in [(-5, 0.5), (0.5, 5), (-4, 13)]:
        assert_allclose(b.integrate(x0, x1), p.integrate(x0, x1))