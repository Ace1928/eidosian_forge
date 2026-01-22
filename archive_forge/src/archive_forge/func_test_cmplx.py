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
def test_cmplx(self):
    b = _make_random_spline()
    t, c, k = b.tck
    cc = c * (1.0 + 3j)
    b = BSpline(t, cc, k)
    b_re = BSpline(t, b.c.real, k)
    b_im = BSpline(t, b.c.imag, k)
    xx = np.linspace(t[k], t[-k - 1], 20)
    assert_allclose(b(xx).real, b_re(xx), atol=1e-14)
    assert_allclose(b(xx).imag, b_im(xx), atol=1e-14)