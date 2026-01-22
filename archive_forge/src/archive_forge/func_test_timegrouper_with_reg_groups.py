from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_timegrouper_with_reg_groups(self):
    df_original = DataFrame({'Branch': 'A A A A A A A B'.split(), 'Buyer': 'Carl Mark Carl Carl Joe Joe Joe Carl'.split(), 'Quantity': [1, 3, 5, 1, 8, 1, 9, 3], 'Date': [datetime(2013, 1, 1, 13, 0), datetime(2013, 1, 1, 13, 5), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 12, 2, 12, 0), datetime(2013, 12, 2, 14, 0)]}).set_index('Date')
    df_sorted = df_original.sort_values(by='Quantity', ascending=False)
    for df in [df_original, df_sorted]:
        expected = DataFrame({'Buyer': 'Carl Joe Mark'.split(), 'Quantity': [10, 18, 3], 'Date': [datetime(2013, 12, 31, 0, 0), datetime(2013, 12, 31, 0, 0), datetime(2013, 12, 31, 0, 0)]}).set_index(['Date', 'Buyer'])
        msg = 'The default value of numeric_only'
        result = df.groupby([Grouper(freq='YE'), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'Buyer': 'Carl Mark Carl Joe'.split(), 'Quantity': [1, 3, 9, 18], 'Date': [datetime(2013, 1, 1, 0, 0), datetime(2013, 1, 1, 0, 0), datetime(2013, 7, 1, 0, 0), datetime(2013, 7, 1, 0, 0)]}).set_index(['Date', 'Buyer'])
        result = df.groupby([Grouper(freq='6MS'), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
    df_original = DataFrame({'Branch': 'A A A A A A A B'.split(), 'Buyer': 'Carl Mark Carl Carl Joe Joe Joe Carl'.split(), 'Quantity': [1, 3, 5, 1, 8, 1, 9, 3], 'Date': [datetime(2013, 10, 1, 13, 0), datetime(2013, 10, 1, 13, 5), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 2, 10, 0), datetime(2013, 10, 2, 12, 0), datetime(2013, 10, 2, 14, 0)]}).set_index('Date')
    df_sorted = df_original.sort_values(by='Quantity', ascending=False)
    for df in [df_original, df_sorted]:
        expected = DataFrame({'Buyer': 'Carl Joe Mark Carl Joe'.split(), 'Quantity': [6, 8, 3, 4, 10], 'Date': [datetime(2013, 10, 1, 0, 0), datetime(2013, 10, 1, 0, 0), datetime(2013, 10, 1, 0, 0), datetime(2013, 10, 2, 0, 0), datetime(2013, 10, 2, 0, 0)]}).set_index(['Date', 'Buyer'])
        result = df.groupby([Grouper(freq='1D'), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df.groupby([Grouper(freq='1ME'), 'Buyer']).sum(numeric_only=True)
        expected = DataFrame({'Buyer': 'Carl Joe Mark'.split(), 'Quantity': [10, 18, 3], 'Date': [datetime(2013, 10, 31, 0, 0), datetime(2013, 10, 31, 0, 0), datetime(2013, 10, 31, 0, 0)]}).set_index(['Date', 'Buyer'])
        tm.assert_frame_equal(result, expected)
        df = df.reset_index()
        result = df.groupby([Grouper(freq='1ME', key='Date'), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        with pytest.raises(KeyError, match="'The grouper name foo is not found'"):
            df.groupby([Grouper(freq='1ME', key='foo'), 'Buyer']).sum()
        df = df.set_index('Date')
        result = df.groupby([Grouper(freq='1ME', level='Date'), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df.groupby([Grouper(freq='1ME', level=0), 'Buyer']).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        with pytest.raises(ValueError, match='The level foo is not valid'):
            df.groupby([Grouper(freq='1ME', level='foo'), 'Buyer']).sum()
        df = df.copy()
        df['Date'] = df.index + offsets.MonthEnd(2)
        result = df.groupby([Grouper(freq='1ME', key='Date'), 'Buyer']).sum(numeric_only=True)
        expected = DataFrame({'Buyer': 'Carl Joe Mark'.split(), 'Quantity': [10, 18, 3], 'Date': [datetime(2013, 11, 30, 0, 0), datetime(2013, 11, 30, 0, 0), datetime(2013, 11, 30, 0, 0)]}).set_index(['Date', 'Buyer'])
        tm.assert_frame_equal(result, expected)
        msg = 'The Grouper cannot specify both a key and a level!'
        with pytest.raises(ValueError, match=msg):
            df.groupby([Grouper(freq='1ME', key='Date', level='Date'), 'Buyer']).sum()
        expected = DataFrame([[31]], columns=['Quantity'], index=DatetimeIndex([datetime(2013, 10, 31, 0, 0)], freq=offsets.MonthEnd(), name='Date'))
        result = df.groupby(Grouper(freq='1ME')).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df.groupby([Grouper(freq='1ME')]).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        expected.index = expected.index.shift(1)
        assert expected.index.freq == offsets.MonthEnd()
        result = df.groupby(Grouper(freq='1ME', key='Date')).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df.groupby([Grouper(freq='1ME', key='Date')]).sum(numeric_only=True)
        tm.assert_frame_equal(result, expected)