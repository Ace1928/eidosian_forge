from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def default_kwargs_base():
    file_path = os.path.join(cur_dir, 'results', 'stl_co2.csv')
    co2 = np.asarray(pd.read_csv(file_path, header=None).iloc[:, 0])
    y = co2
    nobs = y.shape[0]
    nperiod = 12
    work = np.zeros((nobs + 2 * nperiod, 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    return dict(y=y, n=y.shape[0], np=nperiod, ns=35, nt=19, nl=13, no=2, ni=1, nsjump=4, ntjump=2, nljump=2, isdeg=1, itdeg=1, ildeg=1, rw=rw, trend=trend, season=season, work=work)