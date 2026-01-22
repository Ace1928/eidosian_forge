import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_sarimax_nonconsecutive():
    endog = macrodata['infl']
    mod1 = sarimax.SARIMAX(endog, order=([1, 0, 0, 1], 0, 0), enforce_stationarity=False)
    mod2 = sarimax.SARIMAX(endog, order=(4, 0, 0), enforce_stationarity=False)
    start_params = [0.6, 0.2, 6.4]
    res1 = mod1.fit(start_params, disp=False)
    res2 = mod2.fit_constrained({'ar.L2': 0, 'ar.L3': 0}, res1.params, includes_fixed=False, disp=False)
    assert_equal(res1.fixed_params, [])
    assert_equal(res2.fixed_params, ['ar.L2', 'ar.L3'])
    params = np.asarray(res1.params)
    desired = np.r_[params[0], 0, 0, params[1:]]
    assert_allclose(res2.params, desired)
    with mod2.fix_params({'ar.L2': 0, 'ar.L3': 0}):
        res2 = mod2.smooth(res1.params)
    check_results(res1, res2, check_lutkepohl=True)
    with mod2.fix_params({'ar.L2': 0, 'ar.L3': 0}):
        res3 = mod2.filter(res2.params, includes_fixed=True)
        check_results(res1, res3, check_lutkepohl=True)