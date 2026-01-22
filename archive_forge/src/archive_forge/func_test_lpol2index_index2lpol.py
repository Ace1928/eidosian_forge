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
def test_lpol2index_index2lpol():
    process = ArmaProcess([1, 0, 0, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [0.8])
    assert_equal(locs, [2])
    process = ArmaProcess([1, 0.1, 0.1, -0.8])
    coefs, locs = lpol2index(process.arcoefs)
    assert_almost_equal(coefs, [-0.1, -0.1, 0.8])
    assert_equal(locs, [0, 1, 2])
    ar = index2lpol(coefs, locs)
    assert_equal(process.arcoefs, ar)