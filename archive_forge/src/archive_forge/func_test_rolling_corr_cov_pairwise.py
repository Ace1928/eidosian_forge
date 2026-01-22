import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f', ['corr', 'cov'])
def test_rolling_corr_cov_pairwise(self, f, roll_frame):
    g = roll_frame.groupby('A')
    r = g.rolling(window=4)
    result = getattr(r.B, f)(pairwise=True)

    def func(x):
        return getattr(x.B.rolling(4), f)(pairwise=True)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(func)
    tm.assert_series_equal(result, expected)