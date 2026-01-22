import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_results_extend():
    endog = macrodata['infl']
    endog1 = endog.iloc[:100]
    endog2 = endog.iloc[100:]
    mod_full = sarimax.SARIMAX(endog)
    with mod_full.fix_params({'ar.L1': 0.5}):
        res_full = mod_full.smooth([1.0], includes_fixed=False)
    mod = sarimax.SARIMAX(endog1)
    with mod.fix_params({'ar.L1': 0.5}):
        res1 = mod.smooth([1.0], includes_fixed=False)
    res2 = res1.append(endog2)
    assert_allclose(res2.params, res_full.params)
    assert_equal(res2._fixed_params, res_full._fixed_params)
    assert_allclose(res2.llf_obs, res_full.llf_obs)