import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
@pytest.mark.parametrize('missing', ['all', 'partial', 'mixed', None])
@pytest.mark.parametrize('filter_univariate', [True, False])
@pytest.mark.parametrize('tvp', [True, False])
def test_news_basic(missing, filter_univariate, tvp):
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params
    append = np.zeros((10, 2)) * np.nan
    append[0] = [0.1, -0.2]
    endog2 = np.concatenate((mod.endog, append), axis=0)
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    endog3 = endog2.copy()
    endog3[-10:] = np.nan
    mod3 = mod2.clone(endog3)
    res3 = mod3.smooth(params, return_ssm=True)
    for t in [0, 1, 150, mod.nobs - 1, mod.nobs, mod.nobs + 1, mod.nobs + 9]:
        out = res2.news(res, t=t)
        desired = res2.smoothed_forecasts[..., t] - res3.smoothed_forecasts[..., t]
        assert_allclose(out.update_impacts, desired, atol=1e-14)
        assert_equal(out.revision_impacts, None)
        out = res2.news(res, start=t, end=t + 1)
        assert_allclose(out.update_impacts, desired[None, ...], atol=1e-14)