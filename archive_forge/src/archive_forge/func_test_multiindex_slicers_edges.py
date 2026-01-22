from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_multiindex_slicers_edges(self):
    df = DataFrame({'A': ['A0'] * 5 + ['A1'] * 5 + ['A2'] * 5, 'B': ['B0', 'B0', 'B1', 'B1', 'B2'] * 3, 'DATE': ['2013-06-11', '2013-07-02', '2013-07-09', '2013-07-30', '2013-08-06', '2013-06-11', '2013-07-02', '2013-07-09', '2013-07-30', '2013-08-06', '2013-09-03', '2013-10-01', '2013-07-09', '2013-08-06', '2013-09-03'], 'VALUES': [22, 35, 14, 9, 4, 40, 18, 4, 2, 5, 1, 2, 3, 4, 2]})
    df['DATE'] = pd.to_datetime(df['DATE'])
    df1 = df.set_index(['A', 'B', 'DATE'])
    df1 = df1.sort_index()
    result = df1.loc[slice('A1'), :]
    expected = df1.iloc[0:10]
    tm.assert_frame_equal(result, expected)
    result = df1.loc[slice('A2'), :]
    expected = df1
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice(None), slice('B1', 'B2')), :]
    expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13, 14]]
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice(None), slice(None), slice('20130702', '20130709')), :]
    expected = df1.iloc[[1, 2, 6, 7, 12]]
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice('A2'), slice('B0')), :]
    expected = df1.iloc[[0, 1, 5, 6, 10, 11]]
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice(None), slice('B2')), :]
    expected = df1
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice(None), slice('B1', 'B2'), slice('2013-08-06')), :]
    expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13]]
    tm.assert_frame_equal(result, expected)
    result = df1.loc[(slice(None), slice(None), slice('20130701', '20130709')), :]
    expected = df1.iloc[[1, 2, 6, 7, 12]]
    tm.assert_frame_equal(result, expected)