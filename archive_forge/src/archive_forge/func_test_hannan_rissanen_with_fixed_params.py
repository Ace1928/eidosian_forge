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
@pytest.mark.parametrize('ar_order, ma_order, fixed_params', [(1, 1, {}), (1, 1, {'ar.L1': 0}), (2, 3, {'ar.L2': -1, 'ma.L1': 2}), ([0, 1], 0, {'ar.L2': 0}), ([1, 5], [0, 0, 1], {'ar.L5': -10, 'ma.L3': 5})])
def test_hannan_rissanen_with_fixed_params(ar_order, ma_order, fixed_params):
    endog = dta['infl'].diff().iloc[1:101]
    desired_p, _ = hannan_rissanen(endog, ar_order=ar_order, ma_order=ma_order, demean=False, fixed_params=fixed_params)
    mod = ARIMA(endog, order=(ar_order, 0, ma_order), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    with mod.fix_params(fixed_params):
        res = mod.fit(method='hannan_rissanen')
    assert_allclose(res.params, desired_p.params)