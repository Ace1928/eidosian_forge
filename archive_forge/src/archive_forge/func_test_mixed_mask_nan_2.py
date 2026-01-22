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
def test_mixed_mask_nan_2():
    a = [[1, np.nan, 2], [np.nan, np.nan, np.nan], [1, 2, 3], [1, np.nan, 3], [1, np.nan, 3], [1, 2, 3]]
    mask = [[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    a_masked = np.ma.masked_array(a, mask=mask)
    b = [[4, 5, 6]]
    ref1 = stats.ranksums([1, 3], [4, 5, 6])
    ref2 = stats.ranksums([1, 2, 3], [4, 5, 6])
    res = stats.ranksums(a_masked, b, nan_policy='omit', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan, ref1.statistic, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan, ref1.pvalue, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)
    res = stats.ranksums(a_masked, b, nan_policy='propagate', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan, np.nan, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan, np.nan, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)