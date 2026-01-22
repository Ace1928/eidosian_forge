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
def test_detrend_dataframe(self):
    columns = ['one', 'two']
    index = [c for c in 'abcde']
    data = pd.DataFrame(self.data_2d, columns=columns, index=index)
    detrended = tools.detrend(data, order=1, axis=0)
    assert_array_almost_equal(detrended.values, np.zeros_like(data))
    assert_frame_equal(detrended, pd.DataFrame(detrended.values, columns=columns, index=index))
    detrended = tools.detrend(data, order=0, axis=0)
    assert_array_almost_equal(detrended.values, [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]])
    assert_frame_equal(detrended, pd.DataFrame(detrended.values, columns=columns, index=index))
    detrended = tools.detrend(data, order=0, axis=1)
    assert_array_almost_equal(detrended.values, [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
    assert_frame_equal(detrended, pd.DataFrame(detrended.values, columns=columns, index=index))