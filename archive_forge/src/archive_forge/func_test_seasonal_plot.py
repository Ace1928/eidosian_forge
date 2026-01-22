from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.python import lmap
import calendar
from io import BytesIO
import locale
import numpy as np
from numpy.testing import assert_, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import elnino, macrodata
from statsmodels.graphics.tsaplots import (
from statsmodels.tsa import arima_process as tsp
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
@pytest.mark.matplotlib
def test_seasonal_plot(close_figures):
    rs = np.random.RandomState(1234)
    data = rs.randn(20, 12)
    data += 6 * np.sin(np.arange(12.0) / 11 * np.pi)[None, :]
    data = data.ravel()
    months = np.tile(np.arange(1, 13), (20, 1))
    months = months.ravel()
    df = pd.DataFrame([data, months], index=['data', 'months']).T
    grouped = df.groupby('months')['data']
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig = seasonal_plot(grouped, labels)
    ax = fig.get_axes()[0]
    output = [tl.get_text() for tl in ax.get_xticklabels()]
    assert_equal(labels, output)