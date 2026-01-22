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
def test_invertroots(self):
    process1 = ArmaProcess.from_coeffs([], [2.5])
    process2 = process1.invertroots(True)
    assert_almost_equal(process2.ma, np.array([1.0, 0.4]))
    process1 = ArmaProcess.from_coeffs([], [0.4])
    process2 = process1.invertroots(True)
    assert_almost_equal(process2.ma, np.array([1.0, 0.4]))
    process1 = ArmaProcess.from_coeffs([], [2.5])
    roots, invertable = process1.invertroots(False)
    assert_equal(invertable, False)
    assert_almost_equal(roots, np.array([1, 0.4]))