from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_empty_sum():
    df = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c']), 'B': [1, 2, 1]})
    expected_idx = CategoricalIndex(['a', 'b', 'c'], name='A')
    result = df.groupby('A', observed=False).B.sum()
    expected = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=0)
    expected = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=1)
    expected = Series([3, 1, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=2)
    expected = Series([3, np.nan, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)