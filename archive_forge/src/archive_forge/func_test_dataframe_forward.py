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
def test_dataframe_forward(self):
    data = self.macro_df
    columns = list(data.columns)
    n = data.shape[0]
    values = np.zeros((n + 3, 16))
    values[:n, :4] = data.values
    for lag in range(1, 4):
        new_cols = [col + '.L.' + str(lag) for col in data]
        columns.extend(new_cols)
        values[lag:n + lag, 4 * lag:4 * (lag + 1)] = data.values
    index = data.index
    values = values[:n]
    expected = pd.DataFrame(values, columns=columns, index=index)
    both = stattools.lagmat(self.macro_df, 3, trim='forward', original='in', use_pandas=True)
    assert_frame_equal(both, expected)
    lags = stattools.lagmat(self.macro_df, 3, trim='forward', original='ex', use_pandas=True)
    assert_frame_equal(lags, expected.iloc[:, 4:])
    lags, lead = stattools.lagmat(self.macro_df, 3, trim='forward', original='sep', use_pandas=True)
    assert_frame_equal(lags, expected.iloc[:, 4:])
    assert_frame_equal(lead, expected.iloc[:, :4])