import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
def test_get_prediction_memory_conserve():
    endog = dta['infl'].iloc[:20]
    mod1 = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod2 = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod1.ssm.set_conserve_memory(MEMORY_CONSERVE)
    assert_equal(mod1.ssm.conserve_memory, MEMORY_CONSERVE)
    assert_equal(mod2.ssm.conserve_memory, 0)
    res1 = mod1.filter([0])
    res2 = mod2.filter([0])
    assert_equal(res1.filter_results.conserve_memory, MEMORY_CONSERVE)
    assert_equal(res2.filter_results.conserve_memory, MEMORY_NO_SMOOTHING)
    p1 = res1.get_prediction()
    p2 = res2.get_prediction()
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.se_mean, np.nan)
    assert_allclose(p1.conf_int(), np.nan)
    s1 = p1.summary_frame()
    s2 = p2.summary_frame()
    assert_allclose(s1['mean'], s2['mean'])
    assert_allclose(s1.mean_se, np.nan)
    assert_allclose(s1.mean_ci_lower, np.nan)
    assert_allclose(s1.mean_ci_upper, np.nan)