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
def make_interp_full_matr(x, y, t, k):
    """Assemble an spline order k with knots t to interpolate
    y(x) using full matrices.
    Not-a-knot BC only.

    This routine is here for testing only (even though it's functional).
    """
    assert x.size == y.size
    assert t.size == x.size + k + 1
    n = x.size
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        xval = x[j]
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j, left - k:left + 1] = bb
    c = sl.solve(A, y)
    return c