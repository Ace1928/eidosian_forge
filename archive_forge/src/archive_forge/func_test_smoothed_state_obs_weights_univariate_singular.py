import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('singular', ['both', 0, 1])
@pytest.mark.parametrize('periods', [1, 2])
def test_smoothed_state_obs_weights_univariate_singular(singular, periods, reset_randomstate):
    endog = np.zeros((10, 2))
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2) * 0)
    if singular == 'both':
        mod['obs_cov', ..., :periods] = 0
    else:
        mod['obs_cov', 0, 1, :periods] = 0
        mod['obs_cov', 1, 0, :periods] = 0
        mod['obs_cov', singular, singular, :periods] = 0
    mod['state_cov', :, :, :periods] = 0
    mod.ssm.filter_univariate = True
    res = mod.smooth([])
    for i in range(periods):
        eigvals = np.linalg.eigvalsh(res.forecasts_error_cov[..., i])
        assert_equal(np.min(eigvals), 0)
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
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2) * 0)
                tmp_mod.ssm.filter_univariate = True
                tmp_res = tmp_mod.smooth([])
                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    actual, _, _ = tools.compute_smoothed_state_weights(res)
    assert_allclose(actual, desired, atol=1e-12)