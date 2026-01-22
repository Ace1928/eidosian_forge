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
def test_default_trend():
    endog = dta['infl'].iloc[:50]
    mod = ARIMA(endog)
    assert_equal(mod._spec_arima.trend_order, 0)
    assert_allclose(mod.exog, np.ones((mod.nobs, 1)))
    mod = ARIMA(endog, order=(0, 1, 0))
    assert_equal(mod._spec_arima.trend_order, None)
    assert_equal(mod.exog, None)