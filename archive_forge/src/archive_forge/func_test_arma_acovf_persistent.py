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
def test_arma_acovf_persistent():
    ar = np.array([1, -0.9995])
    ma = np.array([1])
    process = ArmaProcess(ar, ma)
    res = process.acovf(10)
    sig2 = 1 / (1 - 0.9995 ** 2)
    corrs = 0.9995 ** np.arange(10)
    expected = sig2 * corrs
    assert_equal(res.ndim, 1)
    assert_allclose(res, expected)