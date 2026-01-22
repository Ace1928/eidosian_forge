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
@pytest.mark.matplotlib
def test_plot_too_few_obs(reset_randomstate):
    mod = sarimax.SARIMAX(np.random.normal(size=10), order=(10, 0, 0), enforce_stationarity=False)
    with pytest.warns(UserWarning, match='Too few'):
        results = mod.fit()
    with pytest.raises(ValueError, match='Length of endogenous'):
        results.plot_diagnostics(figsize=(15, 5))
    y = np.random.standard_normal(9)
    mod = sarimax.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    with pytest.warns(UserWarning, match='Too few'):
        results = mod.fit()
    with pytest.raises(ValueError, match='Length of endogenous'):
        results.plot_diagnostics(figsize=(30, 15))