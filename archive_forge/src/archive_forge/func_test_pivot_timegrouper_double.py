from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_timegrouper_double(self):
    df = DataFrame({'Branch': 'A A A A A A A B'.split(), 'Buyer': 'Carl Mark Carl Carl Joe Joe Joe Carl'.split(), 'Quantity': [1, 3, 5, 1, 8, 1, 9, 3], 'Date': [datetime(2013, 11, 1, 13, 0), datetime(2013, 9, 1, 13, 5), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 11, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 10, 2, 12, 0), datetime(2013, 12, 5, 14, 0)], 'PayDay': [datetime(2013, 10, 4, 0, 0), datetime(2013, 10, 15, 13, 5), datetime(2013, 9, 5, 20, 0), datetime(2013, 11, 2, 10, 0), datetime(2013, 10, 7, 20, 0), datetime(2013, 9, 5, 10, 0), datetime(2013, 12, 30, 12, 0), datetime(2013, 11, 20, 14, 0)]})
    result = pivot_table(df, index=Grouper(freq='ME', key='Date'), columns=Grouper(freq='ME', key='PayDay'), values='Quantity', aggfunc='sum')
    expected = DataFrame(np.array([np.nan, 3, np.nan, np.nan, 6, np.nan, 1, 9, np.nan, 9, np.nan, np.nan, np.nan, np.nan, 3, np.nan]).reshape(4, 4), index=pd.DatetimeIndex([datetime(2013, 9, 30), datetime(2013, 10, 31), datetime(2013, 11, 30), datetime(2013, 12, 31)], freq='ME'), columns=pd.DatetimeIndex([datetime(2013, 9, 30), datetime(2013, 10, 31), datetime(2013, 11, 30), datetime(2013, 12, 31)], freq='ME'))
    expected.index.name = 'Date'
    expected.columns.name = 'PayDay'
    tm.assert_frame_equal(result, expected)
    result = pivot_table(df, index=Grouper(freq='ME', key='PayDay'), columns=Grouper(freq='ME', key='Date'), values='Quantity', aggfunc='sum')
    tm.assert_frame_equal(result, expected.T)
    tuples = [(datetime(2013, 9, 30), datetime(2013, 10, 31)), (datetime(2013, 10, 31), datetime(2013, 9, 30)), (datetime(2013, 10, 31), datetime(2013, 11, 30)), (datetime(2013, 10, 31), datetime(2013, 12, 31)), (datetime(2013, 11, 30), datetime(2013, 10, 31)), (datetime(2013, 12, 31), datetime(2013, 11, 30))]
    idx = MultiIndex.from_tuples(tuples, names=['Date', 'PayDay'])
    expected = DataFrame(np.array([3, np.nan, 6, np.nan, 1, np.nan, 9, np.nan, 9, np.nan, np.nan, 3]).reshape(6, 2), index=idx, columns=['A', 'B'])
    expected.columns.name = 'Branch'
    result = pivot_table(df, index=[Grouper(freq='ME', key='Date'), Grouper(freq='ME', key='PayDay')], columns=['Branch'], values='Quantity', aggfunc='sum')
    tm.assert_frame_equal(result, expected)
    result = pivot_table(df, index=['Branch'], columns=[Grouper(freq='ME', key='Date'), Grouper(freq='ME', key='PayDay')], values='Quantity', aggfunc='sum')
    tm.assert_frame_equal(result, expected.T)