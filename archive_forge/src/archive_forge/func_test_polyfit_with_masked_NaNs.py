import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_polyfit_with_masked_NaNs(self):
    x = np.random.rand(10)
    y = np.random.rand(20).reshape(-1, 2)
    x[0] = np.nan
    y[-1, -1] = np.nan
    x = x.view(MaskedArray)
    y = y.view(MaskedArray)
    x[0] = masked
    y[-1, -1] = masked
    C, R, K, S, D = polyfit(x, y, 3, full=True)
    c, r, k, s, d = np.polyfit(x[1:-1], y[1:-1, :], 3, full=True)
    for a, a_ in zip((C, R, K, S, D), (c, r, k, s, d)):
        assert_almost_equal(a, a_)