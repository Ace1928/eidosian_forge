import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_nan(self):
    df = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]})
    expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 9, 2, np.nan, 5, 5, 4]}, index=[2, 0, 3, 1, 6, 4, 5])
    sorted_df = df.sort_values(['A'], na_position='first')
    tm.assert_frame_equal(sorted_df, expected)
    expected = DataFrame({'A': [np.nan, 8, 6, 4, 2, 1, 1], 'B': [5, 4, 5, 5, np.nan, 9, 2]}, index=[2, 5, 4, 6, 1, 0, 3])
    sorted_df = df.sort_values(['A'], na_position='first', ascending=False)
    tm.assert_frame_equal(sorted_df, expected)
    expected = df.reindex(columns=['B', 'A'])
    sorted_df = df.sort_values(by=1, axis=1, na_position='first')
    tm.assert_frame_equal(sorted_df, expected)
    expected = DataFrame({'A': [1, 1, 2, 4, 6, 8, np.nan], 'B': [2, 9, np.nan, 5, 5, 4, 5]}, index=[3, 0, 1, 6, 4, 5, 2])
    sorted_df = df.sort_values(['A', 'B'])
    tm.assert_frame_equal(sorted_df, expected)
    expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 2, 9, np.nan, 5, 5, 4]}, index=[2, 3, 0, 1, 6, 4, 5])
    sorted_df = df.sort_values(['A', 'B'], na_position='first')
    tm.assert_frame_equal(sorted_df, expected)
    expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 9, 2, np.nan, 5, 5, 4]}, index=[2, 0, 3, 1, 6, 4, 5])
    sorted_df = df.sort_values(['A', 'B'], ascending=[1, 0], na_position='first')
    tm.assert_frame_equal(sorted_df, expected)
    expected = DataFrame({'A': [8, 6, 4, 2, 1, 1, np.nan], 'B': [4, 5, 5, np.nan, 2, 9, 5]}, index=[5, 4, 6, 1, 3, 0, 2])
    sorted_df = df.sort_values(['A', 'B'], ascending=[0, 1], na_position='last')
    tm.assert_frame_equal(sorted_df, expected)