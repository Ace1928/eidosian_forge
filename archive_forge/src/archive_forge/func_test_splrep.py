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
def test_splrep(self):
    x, y = (self.xx, self.yy)
    tck = splrep(x, y)
    t, c, k = _impl.splrep(x, y)
    assert_allclose(tck[0], t, atol=1e-15)
    assert_allclose(tck[1], c, atol=1e-15)
    assert_equal(tck[2], k)
    tck_f, _, _, _ = splrep(x, y, full_output=True)
    assert_allclose(tck_f[0], t, atol=1e-15)
    assert_allclose(tck_f[1], c, atol=1e-15)
    assert_equal(tck_f[2], k)
    yy = splev(x, tck)
    assert_allclose(y, yy, atol=1e-15)
    b = BSpline(*tck)
    assert_allclose(y, b(x), atol=1e-15)