from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
@pytest.mark.smoke
def test_seasonal_decompose_smoke():
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809, 530, 489, 540, 457, 195, 176, 337, 239, 128, 102, 232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    seasonal_decompose(x, period=4)
    data = pd.DataFrame(x, pd.date_range(start='1/1/1951', periods=len(x), freq=QUARTER_END))
    seasonal_decompose(data)