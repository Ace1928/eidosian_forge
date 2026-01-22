import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_dynamic_varmax(use_exog, trend):
    endog = np.log(dta[['realgdp', 'cpi']])
    if trend in ['n', 'c']:
        endog = endog.diff().iloc[1:] * 100
    if trend == 'n':
        endog -= endog.mean()
    exog = np.ones(100) if use_exog else None
    if use_exog:
        exog[20:40] = 2.0
    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1.0, 2.0]
    cov_params = [1.0, 0.0, 1.0]
    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params
    x_fit1 = exog[:50] if use_exog else None
    x_fcast1 = exog[50:61] if use_exog else None
    mod1 = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x_fit1)
    res1 = mod1.filter(params)
    x_fit2 = exog[:20] if use_exog else None
    x_fcast2 = exog[20:61] if use_exog else None
    mod2 = varmax.VARMAX(endog[:20], order=(1, 0), trend=trend, exog=x_fit2)
    res2 = mod2.filter(params)
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1, information_set='filtered')
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1, information_set='filtered')
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1, information_set='filtered')
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1, signal_only=True)
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1, signal_only=True)
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1, signal_only=True)
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1, signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2, signal_only=True, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1, signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2, signal_only=True, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1, signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2, signal_only=True, information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)