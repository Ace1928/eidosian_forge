from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_lagmat2ds_use_pandas(self):
    data = self.macro_df
    lagmat = stattools.lagmat2ds(data, 2, use_pandas=True)
    expected = self._prepare_expected(data.values, 2)
    cols = []
    for c in data:
        for lags in range(3):
            if lags == 0:
                cols.append(c)
            else:
                cols.append(c + '.L.' + str(lags))
    expected = pd.DataFrame(expected, index=data.index, columns=cols)
    assert_frame_equal(lagmat, expected)
    lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, use_pandas=True, trim='both')
    expected = self._prepare_expected(data.values[:, :2], 3)
    cols = []
    for c in data.iloc[:, :2]:
        for lags in range(4):
            if lags == 0:
                cols.append(c)
            else:
                cols.append(c + '.L.' + str(lags))
    expected = pd.DataFrame(expected, index=data.index, columns=cols)
    expected = expected.iloc[3:]
    assert_frame_equal(lagmat, expected)
    data = self.series
    lagmat = stattools.lagmat2ds(data, 5, use_pandas=True)
    expected = self._prepare_expected(data.values[:, None], 5)
    cols = []
    c = data.name
    for lags in range(6):
        if lags == 0:
            cols.append(c)
        else:
            cols.append(c + '.L.' + str(lags))
    expected = pd.DataFrame(expected, index=data.index, columns=cols)
    assert_frame_equal(lagmat, expected)