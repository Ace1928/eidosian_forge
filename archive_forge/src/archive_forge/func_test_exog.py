from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_exog(self):
    data = self.res0.model.endog
    res_lin_trend = VAR(data).fit(maxlags=2, trend='ct')
    ex = np.arange(len(data))
    res_lin_trend1 = VAR(data, exog=ex).fit(maxlags=2)
    ex2 = np.arange(len(data))[:, None] ** [0, 1]
    res_lin_trend2 = VAR(data, exog=ex2).fit(maxlags=2, trend='n')
    assert_allclose(res_lin_trend.params, res_lin_trend1.params, rtol=0.005)
    assert_allclose(res_lin_trend.params, res_lin_trend2.params, rtol=0.005)
    assert_allclose(res_lin_trend1.params, res_lin_trend2.params, rtol=1e-10)
    y1 = res_lin_trend.simulate_var(seed=987128)
    y2 = res_lin_trend1.simulate_var(seed=987128)
    y3 = res_lin_trend2.simulate_var(seed=987128)
    assert_allclose(y2.mean(0), y1.mean(0), rtol=1e-12)
    assert_allclose(y3.mean(0), y1.mean(0), rtol=1e-12)
    assert_allclose(y3.mean(0), y2.mean(0), rtol=1e-12)
    h = 10
    fc1 = res_lin_trend.forecast(res_lin_trend.endog[-2:], h)
    exf = np.arange(len(data), len(data) + h)
    fc2 = res_lin_trend1.forecast(res_lin_trend1.endog[-2:], h, exog_future=exf)
    with pytest.raises(ValueError, match='exog_future only has'):
        wrong_exf = np.arange(len(data), len(data) + h // 2)
        res_lin_trend1.forecast(res_lin_trend1.endog[-2:], h, exog_future=wrong_exf)
    exf2 = exf[:, None] ** [0, 1]
    fc3 = res_lin_trend2.forecast(res_lin_trend2.endog[-2:], h, exog_future=exf2)
    assert_allclose(fc2, fc1, rtol=1e-12, atol=1e-12)
    assert_allclose(fc3, fc1, rtol=1e-12, atol=1e-12)
    assert_allclose(fc3, fc2, rtol=1e-12, atol=1e-12)
    fci1 = res_lin_trend.forecast_interval(res_lin_trend.endog[-2:], h)
    exf = np.arange(len(data), len(data) + h)
    fci2 = res_lin_trend1.forecast_interval(res_lin_trend1.endog[-2:], h, exog_future=exf)
    exf2 = exf[:, None] ** [0, 1]
    fci3 = res_lin_trend2.forecast_interval(res_lin_trend2.endog[-2:], h, exog_future=exf2)
    assert_allclose(fci2, fci1, rtol=1e-12, atol=1e-12)
    assert_allclose(fci3, fci1, rtol=1e-12, atol=1e-12)
    assert_allclose(fci3, fci2, rtol=1e-12, atol=1e-12)