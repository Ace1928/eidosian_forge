from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_direct_holt_add():
    mod = SimpleExpSmoothing(housing_data, initialization_method='estimated')
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    x = np.squeeze(np.asarray(mod.endog))
    alpha = res.params['smoothing_level']
    l, b, f, _, xhat = _simple_dbl_exp_smoother(x, alpha, beta=0.0, l0=res.params['initial_level'], b0=0.0, nforecast=5)
    assert_allclose(l, res.level)
    assert_allclose(f, res.level.iloc[-1] * np.ones(5))
    assert_allclose(f, res.forecast(5))
    mod = ExponentialSmoothing(housing_data, trend='add', initialization_method='estimated')
    res = mod.fit()
    x = np.squeeze(np.asarray(mod.endog))
    alpha = res.params['smoothing_level']
    beta = res.params['smoothing_trend']
    l, b, f, _, xhat = _simple_dbl_exp_smoother(x, alpha, beta=beta, l0=res.params['initial_level'], b0=res.params['initial_trend'], nforecast=5)
    assert_allclose(xhat, res.fittedvalues)
    assert_allclose(l + b, res.level + res.trend)
    assert_allclose(l, res.level)
    assert_allclose(b, res.trend)
    assert_allclose(f, res.level.iloc[-1] + res.trend.iloc[-1] * np.array([1, 2, 3, 4, 5]))
    assert_allclose(f, res.forecast(5))
    assert isinstance(res.summary().as_text(), str)