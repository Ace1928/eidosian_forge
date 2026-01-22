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
def test_rndm_unity(self):
    b = _make_random_spline()
    b.c = np.ones_like(b.c)
    xx = np.linspace(b.t[b.k], b.t[-b.k - 1], 100)
    assert_allclose(b(xx), 1.0)