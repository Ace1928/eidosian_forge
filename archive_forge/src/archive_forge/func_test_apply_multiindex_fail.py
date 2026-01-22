from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_multiindex_fail():
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame({'d': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 'c': np.tile(['a', 'b', 'c'], 2), 'v': np.arange(1.0, 7.0)}, index=index)

    def f(group):
        v = group['v']
        group['v2'] = (v - v.min()) / (v.max() - v.min())
        return group
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('d', group_keys=False).apply(f)
    expected = df.copy()
    expected['v2'] = np.tile([0.0, 0.5, 1], 2)
    tm.assert_frame_equal(result, expected)