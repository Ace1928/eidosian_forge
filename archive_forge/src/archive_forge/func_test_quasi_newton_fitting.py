from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
@pytest.mark.filterwarnings('ignore:Log-likelihood decreased')
def test_quasi_newton_fitting(reset_randomstate):
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=2, nobs=1000)
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1, standardize=False, idiosyncratic_ar1=False)
    mod_dfm_ar1 = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1, standardize=False, idiosyncratic_ar1=True)
    x = mod_dfm_ar1.start_params
    y = mod_dfm_ar1.untransform_params(x)
    z = mod_dfm_ar1.transform_params(y)
    assert_allclose(x, z)
    res_lbfgs = mod_dfm.fit(method='lbfgs')
    params_lbfgs = res_lbfgs.params.copy()
    start_params = params_lbfgs.copy()
    start_params['L1.0->0'] += 0.01
    start_params['fb(0).cov.chol[1,1]'] += 0.01
    res_em = mod_dfm.fit(start_params, em_initialization=False)
    params_em = res_em.params.copy()
    assert_allclose(res_lbfgs.llf, res_em.llf, atol=0.05, rtol=1e-05)
    assert_allclose(params_lbfgs, params_em, atol=0.05, rtol=1e-05)
    res_lbfgs = mod_dfm_ar1.fit(method='lbfgs')
    params_lbfgs = res_lbfgs.params.copy()
    start_params = params_lbfgs.copy()
    start_params['L1.0->0'] += 0.01
    start_params['fb(0).cov.chol[1,1]'] += 0.01
    res_em = mod_dfm_ar1.fit(params_lbfgs, em_initialization=False)
    params_em = res_em.params.copy()
    assert_allclose(res_lbfgs.llf, res_em.llf, atol=0.05, rtol=1e-05)
    assert_allclose(params_lbfgs, params_em, atol=0.05, rtol=1e-05)