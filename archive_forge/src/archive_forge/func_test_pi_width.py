from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
def test_pi_width():
    rs = np.random.RandomState(1233091)
    y = np.arange(100) + rs.standard_normal(100)
    th = ThetaModel(y, period=12, deseasonalize=False)
    res = th.fit()
    pi = res.prediction_intervals(24)
    d = np.squeeze(np.diff(np.asarray(pi), axis=1))
    assert np.all(np.diff(d) > 0)