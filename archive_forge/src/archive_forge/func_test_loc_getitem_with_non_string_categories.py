import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx_values', [[1, 2, 3], [-1, -2, -3], [1.5, 2.5, 3.5], [-1.5, -2.5, -3.5], *(np.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_NUMPY_DTYPES), *(np.array([1.5, 2.5, 3.5], dtype=dtyp) for dtyp in tm.FLOAT_NUMPY_DTYPES), np.array([1, 'b', 3.5], dtype=object), [Interval(1, 4), Interval(4, 6), Interval(6, 9)], [Timestamp(2019, 1, 1), Timestamp(2019, 2, 1), Timestamp(2019, 3, 1)], [Timedelta(1, 'd'), Timedelta(2, 'd'), Timedelta(3, 'D')], *(pd.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES), pd.IntervalIndex.from_breaks([1, 4, 6, 9]).array, pd.date_range('2019-01-01', periods=3).array, pd.timedelta_range(start='1d', periods=3).array])
def test_loc_getitem_with_non_string_categories(self, idx_values, ordered):
    cat_idx = CategoricalIndex(idx_values, ordered=ordered)
    df = DataFrame({'A': ['foo', 'bar', 'baz']}, index=cat_idx)
    sl = slice(idx_values[0], idx_values[1])
    result = df.loc[idx_values[0]]
    expected = Series(['foo'], index=['A'], name=idx_values[0])
    tm.assert_series_equal(result, expected)
    result = df.loc[idx_values[:2]]
    expected = DataFrame(['foo', 'bar'], index=cat_idx[:2], columns=['A'])
    tm.assert_frame_equal(result, expected)
    result = df.loc[sl]
    expected = DataFrame(['foo', 'bar'], index=cat_idx[:2], columns=['A'])
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[idx_values[0]] = 'qux'
    expected = DataFrame({'A': ['qux', 'bar', 'baz']}, index=cat_idx)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[idx_values[:2], 'A'] = ['qux', 'qux2']
    expected = DataFrame({'A': ['qux', 'qux2', 'baz']}, index=cat_idx)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[sl, 'A'] = ['qux', 'qux2']
    expected = DataFrame({'A': ['qux', 'qux2', 'baz']}, index=cat_idx)
    tm.assert_frame_equal(result, expected)