import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('compute_j', [np.arange(10), [0, 1, 2], [5, 0, 9], 8])
@pytest.mark.parametrize('compute_t', [np.arange(10), [3, 2, 2], [0, 2, 5], 5])
def test_compute_t_compute_j(compute_j, compute_t, reset_randomstate):
    endog = np.zeros((10, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod['obs_intercept'] = np.zeros((6, 1))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    mod.ssm.filter_collapsed = True
    res = mod.smooth([])
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog
    desired = np.zeros((n, n, m, p)) * np.nan
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                tmp_mod['obs_intercept'] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])
                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    actual, _, _ = tools.compute_smoothed_state_weights(res, compute_t=compute_t, compute_j=compute_j)
    compute_t = np.unique(np.atleast_1d(compute_t))
    compute_t.sort()
    compute_j = np.unique(np.atleast_1d(compute_j))
    compute_j.sort()
    for t in np.arange(10):
        if t not in compute_t:
            desired[t, :] = np.nan
    for j in np.arange(10):
        if j not in compute_j:
            desired[:, j] = np.nan
    ix = np.ix_(compute_t, compute_j)
    desired = desired[ix]
    assert_allclose(actual, desired, atol=1e-07)