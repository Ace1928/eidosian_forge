import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_ols_noncentrality(self):
    k = self.k_groups
    res_ols = OLS(self.y, self.ex).fit()
    nobs_t = res_ols.model.nobs
    c_equal = -np.eye(k)[1:]
    c_equal[:, 0] = 1
    v = np.zeros(c_equal.shape[0])
    wt = res_ols.wald_test(c_equal, scalar=True)
    df_num, df_denom = (wt.df_num, wt.df_denom)
    cov_p = res_ols.cov_params()
    nc_wt = wald_test_noncent_generic(res_ols.params, c_equal, v, cov_p, diff=None, joint=True)
    assert_allclose(nc_wt, wt.statistic * wt.df_num, rtol=1e-13)
    nc_wt2 = wald_test_noncent(res_ols.params, c_equal, v, res_ols, diff=None, joint=True)
    assert_allclose(nc_wt2, nc_wt, rtol=1e-13)
    es_ols = nc_wt / nobs_t
    es_oneway = smo.effectsize_oneway(res_ols.params, res_ols.scale, self.nobs, use_var='equal')
    assert_allclose(es_ols, es_oneway, rtol=1e-13)
    alpha = 0.05
    pow_ols = smpwr.ftest_power(np.sqrt(es_ols), df_denom, df_num, alpha, ncc=1)
    pow_oneway = smpwr.ftest_anova_power(np.sqrt(es_oneway), nobs_t, alpha, k_groups=k, df=None)
    assert_allclose(pow_ols, pow_oneway, rtol=1e-13)
    params_alt = res_ols.params * 0.75
    v_off = _offset_constraint(c_equal, res_ols.params, params_alt)
    wt_off = res_ols.wald_test((c_equal, v + v_off), scalar=True)
    nc_wt_off = wald_test_noncent_generic(params_alt, c_equal, v, cov_p, diff=None, joint=True)
    assert_allclose(nc_wt_off, wt_off.statistic * wt_off.df_num, rtol=1e-13)
    nc_wt_vec = wald_test_noncent_generic(params_alt, c_equal, v, cov_p, diff=None, joint=False)
    for i in range(c_equal.shape[0]):
        nc_wt_i = wald_test_noncent_generic(params_alt, c_equal[i:i + 1], v[i:i + 1], cov_p, diff=None, joint=False)
        assert_allclose(nc_wt_vec[i], nc_wt_i, rtol=1e-13)