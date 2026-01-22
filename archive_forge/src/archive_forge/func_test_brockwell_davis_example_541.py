import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
@pytest.mark.low_precision('Test against Example 5.4.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_541():
    endog = oshorts.copy()
    initial, _ = innovations(endog, ma_order=1, demean=True)
    p, _ = innovations_mle(endog, order=(0, 0, 1), demean=True, start_params=initial[1].params)
    assert_allclose(p.ma_params, -0.818, atol=0.001)