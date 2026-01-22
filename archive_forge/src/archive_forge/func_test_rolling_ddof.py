import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f', ['std', 'var'])
def test_rolling_ddof(self, f, roll_frame):
    g = roll_frame.groupby('A', group_keys=False)
    r = g.rolling(window=4)
    result = getattr(r, f)(ddof=1)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
    expected = expected.drop('A', axis=1)
    expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
    expected.index = expected_index
    tm.assert_frame_equal(result, expected)