from statsmodels.compat.pandas import MONTH_END
import pandas as pd
import pytest
from statsmodels.datasets import co2, macrodata
from statsmodels.tsa.x13 import (
@pytest.mark.matplotlib
def test_x13_arima_plot(dataset):
    res = x13_arima_analysis(dataset)
    res.plot()