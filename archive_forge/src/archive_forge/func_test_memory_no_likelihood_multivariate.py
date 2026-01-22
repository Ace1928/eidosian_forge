import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
@pytest.mark.parametrize('univariate', [True, False])
@pytest.mark.parametrize('diffuse', [True, False])
def test_memory_no_likelihood_multivariate(univariate, diffuse):
    endog = dta[['infl', 'realint']].iloc[:20].copy()
    endog.iloc[0, 0] = np.nan
    endog.iloc[4:6, :] = np.nan
    exog = np.log(dta['realgdp'].iloc[:20])
    mod = varmax.VARMAX(endog, order=(1, 0), exog=exog, trend='c')
    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True
    params = [1.4, 1.3, 0.1, 0.01, 0.02, 0.3, -0.001, 0.001, 1.0, -0.1, 0.6]
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)
    assert_allclose(res1.llf, res2.llf)