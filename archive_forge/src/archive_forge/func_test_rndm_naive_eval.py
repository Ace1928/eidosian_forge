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
def test_rndm_naive_eval(self):
    b = _make_random_spline()
    t, c, k = b.tck
    xx = np.linspace(t[k], t[-k - 1], 50)
    y_b = b(xx)
    y_n = [_naive_eval(x, t, c, k) for x in xx]
    assert_allclose(y_b, y_n, atol=1e-14)
    y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
    assert_allclose(y_b, y_n2, atol=1e-14)