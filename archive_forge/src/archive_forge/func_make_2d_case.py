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
def make_2d_case(self):
    x = np.arange(6)
    y = x ** 3
    spl = make_interp_spline(x, y, k=3)
    y_1 = x ** 3 + 2 * x
    spl_1 = make_interp_spline(x, y_1, k=3)
    t2 = (spl.t, spl_1.t)
    c2 = spl.c[:, None] * spl_1.c[None, :]
    return (t2, c2, 3)