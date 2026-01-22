from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
def test_groupby_selection_other_methods(df):
    rng = date_range('2014', periods=len(df))
    df.columns.name = 'foo'
    df.index = rng
    g = df.groupby(['A'])[['C']]
    g_exp = df[['C']].groupby(df['A'])
    tm.assert_frame_equal(g.fillna(0), g_exp.fillna(0))
    tm.assert_frame_equal(g.dtypes, g_exp.dtypes)
    tm.assert_frame_equal(g.apply(lambda x: x.sum()), g_exp.apply(lambda x: x.sum()))
    tm.assert_frame_equal(g.resample('D').mean(), g_exp.resample('D').mean())
    tm.assert_frame_equal(g.resample('D').ohlc(), g_exp.resample('D').ohlc())
    tm.assert_frame_equal(g.filter(lambda x: len(x) == 3), g_exp.filter(lambda x: len(x) == 3))