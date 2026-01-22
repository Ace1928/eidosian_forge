from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_numeric_coercion_when_datetime():
    df = DataFrame({'Number': [1, 2], 'Date': ['2017-03-02'] * 2, 'Str': ['foo', 'inf']})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby(['Number']).apply(lambda x: x.iloc[0])
    df.Date = pd.to_datetime(df.Date)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(['Number']).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(result['Str'], expected['Str'])
    df = DataFrame({'A': [10, 20, 30], 'B': ['foo', '3', '4'], 'T': [pd.Timestamp('12:31:22')] * 3})

    def get_B(g):
        return g.iloc[0][['B']]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(get_B)['B']
    expected = df.B
    expected.index = df.A
    tm.assert_series_equal(result, expected)

    def predictions(tool):
        out = Series(index=['p1', 'p2', 'useTime'], dtype=object)
        if 'step1' in list(tool.State):
            out['p1'] = str(tool[tool.State == 'step1'].Machine.values[0])
        if 'step2' in list(tool.State):
            out['p2'] = str(tool[tool.State == 'step2'].Machine.values[0])
            out['useTime'] = str(tool[tool.State == 'step2'].oTime.values[0])
        return out
    df1 = DataFrame({'Key': ['B', 'B', 'A', 'A'], 'State': ['step1', 'step2', 'step1', 'step2'], 'oTime': ['', '2016-09-19 05:24:33', '', '2016-09-19 23:59:04'], 'Machine': ['23', '36L', '36R', '36R']})
    df2 = df1.copy()
    df2.oTime = pd.to_datetime(df2.oTime)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df1.groupby('Key').apply(predictions).p1
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df2.groupby('Key').apply(predictions).p1
    tm.assert_series_equal(expected, result)