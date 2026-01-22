import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
@pytest.mark.low_precision('Test against Example 5.2.4 in Brockwell and Davis (2016)')
@pytest.mark.xfail(reason='Suspicious result reported in Brockwell and Davis (2016).')
def test_brockwell_davis_example_524_variance():
    endog = dowj.diff().iloc[1:]
    initial, _ = burg(endog, ar_order=1, demean=True)
    p, _ = innovations_mle(endog, order=(1, 0, 0), demean=True, start_params=initial.params)
    assert_allclose(p.sigma2, 0.02117, atol=0.0001)