import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_misc_exog():
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))
    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)
    models = [sarimax.SARIMAX(endog, exog=exog1, order=(1, 1, 0)), sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0)), sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0), simple_differencing=False), sarimax.SARIMAX(endog_pd, exog=exog1_pd, order=(1, 1, 0)), sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0)), sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0), simple_differencing=False)]
    for mod in models:
        mod.start_params
        res = mod.fit(disp=False)
        res.summary()
        res.predict()
        res.predict(dynamic=True)
        res.get_prediction()
        oos_exog = np.random.normal(size=(1, mod.k_exog))
        res.forecast(steps=1, exog=oos_exog)
        res.get_forecast(steps=1, exog=oos_exog)
        oos_exog = np.random.normal(size=(2, mod.k_exog))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)
        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)
    assert_raises(ValueError, sarimax.SARIMAX, endog, exog=np.zeros((10, 4)), order=(1, 1, 0))