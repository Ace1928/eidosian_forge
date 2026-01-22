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
@pytest.mark.parametrize('extrapolate', [False, True, 'periodic'])
@pytest.mark.parametrize('degree', range(5))
def test_design_matrix_same_as_BSpline_call(self, extrapolate, degree):
    """Test that design_matrix(x) is equivalent to BSpline(..)(x)."""
    np.random.seed(1234)
    x = np.random.random_sample(10 * (degree + 1))
    xmin, xmax = (np.amin(x), np.amax(x))
    k = degree
    t = np.r_[np.linspace(xmin - 2, xmin - 1, degree), np.linspace(xmin, xmax, 2 * (degree + 1)), np.linspace(xmax + 1, xmax + 2, degree)]
    c = np.eye(len(t) - k - 1)
    bspline = BSpline(t, c, k, extrapolate)
    assert_allclose(bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray())
    x = np.array([xmin - 10, xmin - 1, xmax + 1.5, xmax + 10])
    if not extrapolate:
        with pytest.raises(ValueError):
            BSpline.design_matrix(x, t, k, extrapolate)
    else:
        assert_allclose(bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray())