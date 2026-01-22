from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
@pytest.mark.parametrize('d', [0, 1])
@pytest.mark.parametrize('seasonal', [True])
def test_from_estimation(d, seasonal):
    ar = [0.8] if not seasonal else [0.8, 0, 0, 0.2, -0.16]
    ma = [0.4] if not seasonal else [0.4, 0, 0, 0.2, -0.08]
    ap = ArmaProcess.from_coeffs(ar, ma, 500)
    idx = pd.date_range(dt.datetime(1900, 1, 1), periods=500, freq=QUARTER_END)
    data = ap.generate_sample(500)
    if d == 1:
        data = np.cumsum(data)
    data = pd.Series(data, index=idx)
    seasonal_order = (1, 0, 1, 4) if seasonal else None
    mod = ARIMA(data, order=(1, d, 1), seasonal_order=seasonal_order)
    res = mod.fit()
    ap_from = ArmaProcess.from_estimation(res)
    shape = (5,) if seasonal else (1,)
    assert ap_from.arcoefs.shape == shape
    assert ap_from.macoefs.shape == shape