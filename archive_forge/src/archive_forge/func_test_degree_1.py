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
def test_degree_1(self):
    t = [0, 1, 2, 3, 4]
    c = [1, 2, 3]
    k = 1
    b = BSpline(t, c, k)
    x = np.linspace(1, 3, 50)
    assert_allclose(c[0] * B_012(x) + c[1] * B_012(x - 1) + c[2] * B_012(x - 2), b(x), atol=1e-14)
    assert_allclose(splev(x, (t, c, k)), b(x), atol=1e-14)