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
def test_innovations():
    endog = dta['infl'].iloc[:50]
    desired_p, _ = innovations(endog, ma_order=2, demean=False)
    mod = ARIMA(endog, order=(0, 0, 2), trend='n')
    res = mod.fit(method='innovations')
    assert_allclose(res.params, desired_p[-1].params)