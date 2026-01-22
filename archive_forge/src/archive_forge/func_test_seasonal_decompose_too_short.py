from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_seasonal_decompose_too_short(reset_randomstate):
    dates = pd.date_range('2000-01-31', periods=4, freq=QUARTER_END)
    y = np.sin(np.arange(4) / 4 * 2 * np.pi)
    y += np.random.standard_normal(y.size)
    y = pd.Series(y, name='y', index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)
    dates = pd.date_range('2000-01-31', periods=12, freq=MONTH_END)
    y = np.sin(np.arange(12) / 12 * 2 * np.pi)
    y += np.random.standard_normal(y.size)
    y = pd.Series(y, name='y', index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)
    with pytest.raises(ValueError):
        seasonal_decompose(y.values, period=12)