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
def test_gnpdef_case(self):
    mdlfile = os.path.join(self.run_dir, 'gnpdef.csv')
    mdl = np.asarray(pd.read_csv(mdlfile))
    res = zivot_andrews(mdl, maxlag=8, regression='c', autolag='t-stat')
    assert_allclose([res[0], res[1], res[3], res[4]], [-4.12155, 0.28024, 5, 40], rtol=0.001)