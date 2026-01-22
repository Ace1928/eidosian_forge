from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
def test_grangercausality(self):
    mdata = macrodata.load_pandas().data
    mdata = mdata[['realgdp', 'realcons']].values
    data = mdata.astype(float)
    data = np.diff(np.log(data), axis=0)
    r_result = [0.243097, 0.7844328, 195, 2]
    with pytest.warns(FutureWarning, match='verbose is'):
        gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
    assert_almost_equal(r_result, gr[2][0]['ssr_ftest'], decimal=7)
    assert_almost_equal(gr[2][0]['params_ftest'], gr[2][0]['ssr_ftest'], decimal=7)