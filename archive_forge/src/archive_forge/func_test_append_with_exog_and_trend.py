from statsmodels.compat.platform import PLATFORM_WIN32
import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
def test_append_with_exog_and_trend():
    endog = dta['infl'].iloc[:100].values
    exog = np.arange(len(endog)) ** 2
    mod = ARIMA(endog[:50], exog=exog[:50], trend='ct')
    res = mod.fit()
    res_e = res.append(endog[50:], exog=exog[50:])
    mod2 = ARIMA(endog, exog=exog, trend='ct')
    res2 = mod2.filter(res_e.params)
    assert_allclose(res2.llf, res_e.llf)