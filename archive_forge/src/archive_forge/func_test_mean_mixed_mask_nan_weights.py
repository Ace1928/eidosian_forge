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
@pytest.mark.parametrize('weighted_fun_name', ['gmean', 'hmean', 'pmean'])
def test_mean_mixed_mask_nan_weights(weighted_fun_name):
    if weighted_fun_name == 'pmean':

        def weighted_fun(a, **kwargs):
            return stats.pmean(a, p=0.42, **kwargs)
    else:
        weighted_fun = getattr(stats, weighted_fun_name)
    m, n = (3, 20)
    axis = -1
    rng = np.random.default_rng(6541968121)
    a = rng.uniform(size=(m, n))
    b = rng.uniform(size=(m, n))
    mask_a1 = rng.uniform(size=(m, n)) < 0.2
    mask_a2 = rng.uniform(size=(m, n)) < 0.1
    mask_b1 = rng.uniform(size=(m, n)) < 0.15
    mask_b2 = rng.uniform(size=(m, n)) < 0.15
    mask_a1[2, :] = True
    a_nans = a.copy()
    b_nans = b.copy()
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan
    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan
    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan
    a_masked3 = np.ma.masked_array(a, mask=mask_a1 | mask_a2)
    b_masked3 = np.ma.masked_array(b, mask=mask_b1 | mask_b2)
    mask_all = mask_a1 | mask_a2 | mask_b1 | mask_b2
    a_masked4 = np.ma.masked_array(a, mask=mask_all)
    b_masked4 = np.ma.masked_array(b, mask=mask_all)
    with np.testing.suppress_warnings() as sup:
        message = 'invalid value encountered'
        sup.filter(RuntimeWarning, message)
        res = weighted_fun(a_nans, weights=b_nans, nan_policy='omit', axis=axis)
        res1 = weighted_fun(a_masked1, weights=b_masked1, nan_policy='omit', axis=axis)
        res2 = weighted_fun(a_masked2, weights=b_masked2, nan_policy='omit', axis=axis)
        res3 = weighted_fun(a_masked3, weights=b_masked3, nan_policy='raise', axis=axis)
        res4 = weighted_fun(a_masked3, weights=b_masked3, nan_policy='propagate', axis=axis)
        if weighted_fun_name not in {'pmean', 'gmean'}:
            weighted_fun_ma = getattr(stats.mstats, weighted_fun_name)
            res5 = weighted_fun_ma(a_masked4, weights=b_masked4, axis=axis, _no_deco=True)
    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)
    if weighted_fun_name not in {'pmean', 'gmean'}:
        np.testing.assert_allclose(res5.compressed(), res[~np.isnan(res)])