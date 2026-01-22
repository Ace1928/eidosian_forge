import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def check_concentrated_scale(filter_univariate=False, missing=False, **kwargs):
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=index)
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc']]
    if missing:
        endog.iloc[0, 0] = np.nan
        endog.iloc[3:5, :] = np.nan
        endog.iloc[8, 1] = np.nan
    kwargs.update({'tolerance': 0})
    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_conc = varmax.VARMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True
    mod_orig.ssm.filter_univariate = filter_univariate
    mod_conc.ssm.filter_univariate = filter_univariate
    conc_params = mod_conc.start_params
    start_scale = conc_params[mod_conc._params_state_cov][0]
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        conc_params[mod_conc._params_state_cov] /= start_scale
    else:
        conc_params[mod_conc._params_state_cov] /= start_scale ** 0.5
    res_conc = mod_conc.smooth(conc_params)
    scale = res_conc.scale
    orig_params = conc_params.copy()
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        orig_params[mod_orig._params_state_cov] *= scale
    else:
        orig_params[mod_orig._params_state_cov] *= scale ** 0.5
    orig_params[mod_orig._params_obs_cov] *= scale
    res_orig = mod_orig.smooth(orig_params)
    assert_allclose(res_conc.llf, res_orig.llf)
    for name in mod_conc.ssm.shapes:
        if name == 'obs':
            continue
        assert_allclose(getattr(res_conc.filter_results, name), getattr(res_orig.filter_results, name))
    scale = res_conc.scale
    d = res_conc.loglikelihood_burn
    filter_attr = ['predicted_state', 'filtered_state', 'forecasts', 'forecasts_error', 'kalman_gain']
    for name in filter_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-07)
    filter_attr_burn = ['standardized_forecasts_error', 'predicted_state_cov', 'filtered_state_cov', 'tmp1', 'tmp2', 'tmp3', 'tmp4']
    for name in filter_attr_burn:
        actual = getattr(res_conc.filter_results, name)[..., d:]
        desired = getattr(res_orig.filter_results, name)[..., d:]
        assert_allclose(actual, desired, atol=1e-07)
    smoothed_attr = ['smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_state_disturbance', 'smoothed_state_disturbance_cov', 'smoothed_measurement_disturbance', 'smoothed_measurement_disturbance_cov', 'scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothing_error', 'smoothed_forecasts', 'smoothed_forecasts_error', 'smoothed_forecasts_error_cov']
    for name in smoothed_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-07)
    nobs = mod_conc.nobs
    pred_conc = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)
    pred_orig = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)
    assert_allclose(pred_conc.predicted_mean, pred_orig.predicted_mean)
    assert_allclose(pred_conc.se_mean, pred_orig.se_mean)