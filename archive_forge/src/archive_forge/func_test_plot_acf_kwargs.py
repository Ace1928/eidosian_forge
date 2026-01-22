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
def test_plot_acf_kwargs(close_figures):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    buff = BytesIO()
    plot_acf(acf, ax=ax)
    fig.savefig(buff, format='rgba')
    buff_with_vlines = BytesIO()
    fig_with_vlines = plt.figure()
    ax = fig_with_vlines.add_subplot(111)
    vlines_kwargs = {'linestyles': 'dashdot'}
    plot_acf(acf, ax=ax, vlines_kwargs=vlines_kwargs)
    fig_with_vlines.savefig(buff_with_vlines, format='rgba')
    buff.seek(0)
    buff_with_vlines.seek(0)
    plain = buff.read()
    with_vlines = buff_with_vlines.read()
    assert_(with_vlines != plain)