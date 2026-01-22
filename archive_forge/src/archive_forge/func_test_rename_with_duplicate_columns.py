from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_with_duplicate_columns(self):
    df4 = DataFrame({'RT': [0.0454], 'TClose': [22.02], 'TExg': [0.0422]}, index=MultiIndex.from_tuples([(600809, 20130331)], names=['STK_ID', 'RPT_Date']))
    df5 = DataFrame({'RPT_Date': [20120930, 20121231, 20130331], 'STK_ID': [600809] * 3, 'STK_Name': ['饡驦', '饡驦', '饡驦'], 'TClose': [38.05, 41.66, 30.01]}, index=MultiIndex.from_tuples([(600809, 20120930), (600809, 20121231), (600809, 20130331)], names=['STK_ID', 'RPT_Date']))
    k = merge(df4, df5, how='inner', left_index=True, right_index=True)
    result = k.rename(columns={'TClose_x': 'TClose', 'TClose_y': 'QT_Close'})
    expected = DataFrame([[0.0454, 22.02, 0.0422, 20130331, 600809, '饡驦', 30.01]], columns=['RT', 'TClose', 'TExg', 'RPT_Date', 'STK_ID', 'STK_Name', 'QT_Close']).set_index(['STK_ID', 'RPT_Date'], drop=False)
    tm.assert_frame_equal(result, expected)