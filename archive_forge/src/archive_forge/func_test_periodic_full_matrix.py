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
def test_periodic_full_matrix(self):
    k = 3
    b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
    t = _periodic_knots(self.xx, k)
    c = _make_interp_per_full_matr(self.xx, self.yy, t, k)
    b1 = np.vectorize(lambda x: _naive_eval(x, t, c, k))
    assert_allclose(b(self.xx), b1(self.xx), atol=1e-14)