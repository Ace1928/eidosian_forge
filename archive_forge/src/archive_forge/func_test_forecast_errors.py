from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
def test_forecast_errors(data):
    res = ThetaModel(data, period=12).fit()
    with pytest.raises(ValueError, match='steps must be a positive integer'):
        res.forecast(-1)
    with pytest.raises(ValueError, match='theta must be a float'):
        res.forecast(7, theta=0.99)
    with pytest.raises(ValueError, match='steps must be a positive integer'):
        res.forecast_components(0)