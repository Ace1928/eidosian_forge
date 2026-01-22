from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
@pytest.mark.parametrize('axis', range(-3, 3))
def test_masked_stat_3d(axis):
    np.random.seed(0)
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(4, 5)
    c = np.random.rand(4, 1)
    mask_a = a < 0.1
    mask_c = [False, False, False, True]
    a_masked = np.ma.masked_array(a, mask=mask_a)
    c_masked = np.ma.masked_array(c, mask=mask_c)
    a_nans = a.copy()
    a_nans[mask_a] = np.nan
    c_nans = c.copy()
    c_nans[mask_c] = np.nan
    res = stats.kruskal(a_nans, b, c_nans, nan_policy='omit', axis=axis)
    res2 = stats.kruskal(a_masked, b, c_masked, axis=axis)
    np.testing.assert_array_equal(res, res2)