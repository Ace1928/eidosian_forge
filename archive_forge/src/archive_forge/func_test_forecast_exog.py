import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_forecast_exog():
    nobs = 100
    endog = np.ones((nobs, 2)) * 2.0
    exog = np.ones(nobs)
    mod = dynamic_factor.DynamicFactor(endog, exog=exog, k_factors=1, factor_order=1)
    res = mod.smooth(np.r_[[0] * 2, 2.0, 2.0, 1, 1.0, 0.0])
    exog_fcast_scalar = 1.0
    exog_fcast_1dim = np.ones(1)
    exog_fcast_2dim = np.ones((1, 1))
    assert_allclose(res.forecast(1, exog=exog_fcast_scalar), 2.0)
    assert_allclose(res.forecast(1, exog=exog_fcast_1dim), 2.0)
    assert_allclose(res.forecast(1, exog=exog_fcast_2dim), 2.0)
    h = 10
    exog_fcast_1dim = np.ones(h)
    exog_fcast_2dim = np.ones((h, 1))
    assert_allclose(res.forecast(h, exog=exog_fcast_1dim), 2.0)
    assert_allclose(res.forecast(h, exog=exog_fcast_2dim), 2.0)
    assert_raises(ValueError, res.forecast, h, exog=1.0)
    assert_raises(ValueError, res.forecast, h, exog=[1, 2])
    assert_raises(ValueError, res.forecast, h, exog=np.ones((h, 2)))