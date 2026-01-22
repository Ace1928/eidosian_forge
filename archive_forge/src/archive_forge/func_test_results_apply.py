import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_results_apply():
    endog = macrodata['infl']
    mod = sarimax.SARIMAX(endog)
    with mod.fix_params({'ar.L1': 0.5}):
        res = mod.smooth([1.0], includes_fixed=False)
        start_params = [10.3]
        res_fit = mod.fit(start_params, disp=False)
    res2 = res.apply(endog)
    res2_fit = res.apply(endog, refit=True, fit_kwargs={'disp': False, 'start_params': res_fit.params})
    assert_allclose(res2.params, res.params)
    assert_equal(res2._fixed_params, res._fixed_params)
    assert_allclose(res2.llf_obs, res.llf_obs)
    assert_allclose(res2_fit.params, res_fit.params)
    assert_equal(res2_fit._fixed_params, res_fit._fixed_params)
    assert_allclose(res2_fit.llf_obs, res_fit.llf_obs)