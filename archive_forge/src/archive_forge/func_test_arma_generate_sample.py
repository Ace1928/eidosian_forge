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
@pytest.mark.parametrize('ar', arlist)
@pytest.mark.parametrize('ma', malist)
@pytest.mark.parametrize('dist', [np.random.standard_normal])
def test_arma_generate_sample(dist, ar, ma):
    T = 100
    np.random.seed(1234)
    eta = dist(T)
    np.random.seed(1234)
    rep1 = arma_generate_sample(ar, ma, T, distrvs=dist)
    ar_params = -1 * np.array(ar[1:])
    ma_params = np.array(ma[1:])
    rep2 = _manual_arma_generate_sample(ar_params, ma_params, eta)
    assert_array_almost_equal(rep1, rep2, 13)