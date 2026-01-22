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
@pytest.mark.parametrize('k', [2, 3, 4, 5])
def test_periodic_splev(self, k):
    b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
    tck = splrep(self.xx, self.yy, per=True, k=k)
    spl = splev(self.xx, tck)
    assert_allclose(spl, b(self.xx), atol=1e-14)
    for i in range(1, k):
        spl = splev(self.xx, tck, der=i)
        assert_allclose(spl, b(self.xx, nu=i), atol=1e-10)