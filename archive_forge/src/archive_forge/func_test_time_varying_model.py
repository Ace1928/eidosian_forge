import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
def test_time_varying_model(reset_randomstate):
    endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
    np.random.seed(1234)
    mod_switch = TVSS(endog)
    mod_switch['design', ..., 3] = 0
    mod_switch['obs_cov', ..., 3] = 0
    mod_switch['obs_cov', 1, 1, 3] = 1.0
    res_switch = mod_switch.ssm.smooth()
    kfilter = mod_switch.ssm._kalman_filter
    uf_switch = np.array(kfilter.univariate_filter, copy=True)
    np.random.seed(1234)
    mod_uv = TVSS(endog)
    mod_uv['design', ..., 3] = 0
    mod_uv['obs_cov', ..., 3] = 0
    mod_uv['obs_cov', 1, 1, 3] = 1.0
    mod_uv.ssm.filter_univariate = True
    res_uv = mod_uv.ssm.smooth()
    kfilter = mod_uv.ssm._kalman_filter
    uf_uv = np.array(kfilter.univariate_filter, copy=True)
    np.random.seed(1234)
    endog_mv = endog.copy()
    endog_mv[3, 0] = np.nan
    mod_mv = TVSS(endog_mv)
    mod_mv['design', ..., 3] = 0
    mod_mv['obs_cov', ..., 3] = 0
    mod_mv['obs_cov', 1, 1, 3] = 1.0
    res_mv = mod_mv.ssm.smooth()
    kfilter = mod_mv.ssm._kalman_filter
    uf_mv = np.array(kfilter.univariate_filter, copy=True)
    assert_allclose(uf_switch[:3], 0)
    assert_allclose(uf_switch[3], 1)
    assert_allclose(uf_switch[4:], 0)
    assert_allclose(uf_uv, 1)
    assert_allclose(uf_mv, 0)
    check_filter_output([res_mv, res_switch, res_uv], np.s_[3])
    check_smoother_output([res_mv, res_switch, res_uv], np.s_[3])