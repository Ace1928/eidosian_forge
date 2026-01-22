import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_predicted_filtered_smoothed_with_nans_TVSS(reset_randomstate):
    mod = TVSS(np.zeros((50, 2)) * np.nan)
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    res = mod.smooth([])
    mod_oos = TVSS(np.zeros((11, 2)) * np.nan)
    kwargs = {key: mod_oos[key] for key in ['obs_intercept', 'design', 'obs_cov', 'transition', 'selection', 'state_cov']}
    p_pred = res.get_prediction(start=0, end=60, information_set='predicted', **kwargs)
    f_pred = res.get_prediction(start=0, end=60, information_set='filtered', **kwargs)
    s_pred = res.get_prediction(start=0, end=60, information_set='smoothed', **kwargs)
    assert_allclose(s_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(s_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(f_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(f_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(p_pred.predicted_mean[:50], res.fittedvalues)
    assert_allclose(p_pred.var_pred_mean[:50].T, res.forecasts_error_cov)
    p_signal = res.get_prediction(start=0, end=60, information_set='predicted', signal_only=True, **kwargs)
    f_signal = res.get_prediction(start=0, end=60, information_set='filtered', signal_only=True, **kwargs)
    s_signal = res.get_prediction(start=0, end=60, information_set='smoothed', signal_only=True, **kwargs)
    assert_allclose(s_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(s_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(f_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(f_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(p_signal.predicted_mean[:50] + mod['obs_intercept'].T, res.fittedvalues)
    assert_allclose((p_signal.var_pred_mean[:50] + mod['obs_cov'].T).T, res.forecasts_error_cov)