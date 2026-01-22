from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_fill(self, future_stack):
    data = Series([1, 2, 4, 5], dtype=np.int16)
    data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = data.unstack(fill_value=-1)
    expected = DataFrame({'a': [1, -1, 5], 'b': [2, 4, -1]}, index=['x', 'y', 'z'], dtype=np.int16)
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value=0.5)
    expected = DataFrame({'a': [1, 0.5, 5], 'b': [2, 4, 0.5]}, index=['x', 'y', 'z'], dtype=float)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'x': ['a', 'a', 'b'], 'y': ['j', 'k', 'j'], 'z': [0, 1, 2], 'w': [0, 1, 2]}).set_index(['x', 'y', 'z'])
    unstacked = df.unstack(['x', 'y'], fill_value=0)
    key = ('w', 'b', 'j')
    expected = unstacked[key]
    result = Series([0, 0, 2], index=unstacked.index, name=key)
    tm.assert_series_equal(result, expected)
    stacked = unstacked.stack(['x', 'y'], future_stack=future_stack)
    stacked.index = stacked.index.reorder_levels(df.index.names)
    stacked = stacked.astype(np.int64)
    result = stacked.loc[df.index]
    tm.assert_frame_equal(result, df)
    s = df['w']
    result = s.unstack(['x', 'y'], fill_value=0)
    expected = unstacked['w']
    tm.assert_frame_equal(result, expected)