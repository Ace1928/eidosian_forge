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
@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_incompatible_x_y(self, k):
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4, 5, 6, 7]
    with assert_raises(ValueError, match='Shapes of x'):
        make_interp_spline(x, y, k=k)