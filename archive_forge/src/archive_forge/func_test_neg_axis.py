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
def test_neg_axis(self):
    k = 2
    t = [0, 1, 2, 3, 4, 5, 6]
    c = np.array([[-1, 2, 0, -1], [2, 0, -3, 1]])
    spl = BSpline(t, c, k, axis=-1)
    spl0 = BSpline(t, c[0], k)
    spl1 = BSpline(t, c[1], k)
    assert_equal(spl(2.5), [spl0(2.5), spl1(2.5)])