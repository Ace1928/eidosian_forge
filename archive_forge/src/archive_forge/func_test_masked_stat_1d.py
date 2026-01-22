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
def test_masked_stat_1d():
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]
    res = stats.mannwhitneyu(males, females)
    females2 = [20, 11, 17, np.nan, 12]
    res2 = stats.mannwhitneyu(males, females2, nan_policy='omit')
    np.testing.assert_array_equal(res2, res)
    females3 = [20, 11, 17, 1000, 12]
    mask3 = [False, False, False, True, False]
    females3 = np.ma.masked_array(females3, mask=mask3)
    res3 = stats.mannwhitneyu(males, females3)
    np.testing.assert_array_equal(res3, res)
    females4 = [20, 11, 17, np.nan, 1000, 12]
    mask4 = [False, False, False, False, True, False]
    females4 = np.ma.masked_array(females4, mask=mask4)
    res4 = stats.mannwhitneyu(males, females4, nan_policy='omit')
    np.testing.assert_array_equal(res4, res)
    females5 = [20, 11, 17, np.nan, 1000, 12]
    mask5 = [False, False, False, True, True, False]
    females5 = np.ma.masked_array(females5, mask=mask5)
    res5 = stats.mannwhitneyu(males, females5, nan_policy='propagate')
    res6 = stats.mannwhitneyu(males, females5, nan_policy='raise')
    np.testing.assert_array_equal(res5, res)
    np.testing.assert_array_equal(res6, res)