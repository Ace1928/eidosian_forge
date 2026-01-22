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
def test_default_extrap(self):
    b = _make_random_spline()
    t, _, k = b.tck
    xx = [t[0] - 1, t[-1] + 1]
    yy = b(xx)
    assert_(not np.all(np.isnan(yy)))