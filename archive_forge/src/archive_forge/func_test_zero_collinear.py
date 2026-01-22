from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
def test_zero_collinear(self):
    if isinstance(self.results.model, sm.GEE):
        pytest.skip('Not completely generic yet')
    use_start_params = not isinstance(self.results.model, (sm.RLM, sm.OLS, sm.WLS, sm.GLM))
    self.use_start_params = use_start_params
    keep_index = list(range(self.results.model.exog.shape[1]))
    keep_index_p = list(range(self.results.params.shape[0]))
    drop_index = []
    for i in drop_index:
        del keep_index[i]
        del keep_index_p[i]
    keep_index_p = list(range(self.results.params.shape[0]))
    mod2 = self.results.model
    mod_cls = mod2.__class__
    init_kwds = mod2._get_init_kwds()
    ex = np.column_stack((mod2.exog, mod2.exog))
    mod = mod_cls(mod2.endog, ex, **init_kwds)
    keep_index = list(range(self.results.model.exog.shape[1]))
    keep_index_p = list(range(self.results.model.exog.shape[1]))
    k_vars = ex.shape[1]
    k_extra = 0
    if hasattr(mod, 'k_extra') and mod.k_extra > 0:
        keep_index_p += list(range(k_vars, k_vars + mod.k_extra))
        k_extra = mod.k_extra
    warn_cls = HessianInversionWarning if isinstance(mod, sm.GLM) else None
    cov_types = ['nonrobust', 'HC0']
    for cov_type in cov_types:
        if cov_type != 'nonrobust' and isinstance(self.results.model, sm.RLM):
            return
        if use_start_params:
            start_params = np.zeros(k_vars + k_extra)
            method = self.results.mle_settings['optimizer']
            sp = self.results.mle_settings['start_params'].copy()
            if self.transform_index is not None:
                sp[self.transform_index] = np.exp(sp[self.transform_index])
            start_params[keep_index_p] = sp
            with pytest_warns(warn_cls):
                res1 = mod._fit_collinear(cov_type=cov_type, start_params=start_params, method=method, disp=0)
            if cov_type != 'nonrobust':
                with pytest_warns(warn_cls):
                    res2 = self.results.model.fit(cov_type=cov_type, start_params=sp, method=method, disp=0)
        else:
            with pytest_warns(warn_cls):
                if isinstance(self.results.model, sm.RLM):
                    res1 = mod._fit_collinear()
                else:
                    res1 = mod._fit_collinear(cov_type=cov_type)
            if cov_type != 'nonrobust':
                res2 = self.results.model.fit(cov_type=cov_type)
        if cov_type == 'nonrobust':
            res2 = self.results
        if hasattr(res2, 'mle_settings'):
            assert_equal(res1.results_constrained.mle_settings['optimizer'], res2.mle_settings['optimizer'])
            if 'start_params' in res2.mle_settings:
                spc = res1.results_constrained.mle_settings['start_params']
                assert_allclose(spc, res2.mle_settings['start_params'], rtol=1e-10, atol=1e-20)
                assert_equal(res1.mle_settings['optimizer'], res2.mle_settings['optimizer'])
                assert_allclose(res1.mle_settings['start_params'], res2.mle_settings['start_params'], rtol=1e-10, atol=1e-20)
        assert_allclose(res1.params[keep_index_p], res2.params, rtol=1e-06)
        assert_allclose(res1.params[drop_index], 0, rtol=1e-10)
        assert_allclose(res1.bse[keep_index_p], res2.bse, rtol=1e-08)
        assert_allclose(res1.bse[drop_index], 0, rtol=1e-10)
        tvals1 = res1.tvalues[keep_index_p]
        assert_allclose(tvals1, res2.tvalues, rtol=5e-08)
        if PLATFORM_LINUX32 or SCIPY_GT_14:
            pvals1 = res1.pvalues[keep_index_p]
        else:
            pvals1 = res1.pvalues[keep_index_p]
        assert_allclose(pvals1, res2.pvalues, rtol=1e-06, atol=1e-30)
        if hasattr(res1, 'resid'):
            assert_allclose(res1.resid, res2.resid, rtol=1e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        predicted1 = res1.predict(ex, **self.predict_kwds)
        predicted2 = res2.predict(ex[keep_index], **self.predict_kwds)
        assert_allclose(predicted1, predicted2, rtol=1e-08, atol=1e-11)
        ex = res1.model.exog[:5]
        kwds = getattr(self, 'predict_kwds_5', {})
        predicted1 = res1.predict(ex, **kwds)
        predicted2 = res2.predict(ex[:, keep_index], **kwds)
        assert_allclose(predicted1, predicted2, rtol=1e-08, atol=1e-11)