import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_dict_parameter_cast_result_dtypes():
    df = DataFrame({'class': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], 'time': date_range('1/1/2011', periods=8, freq='h')})
    df.loc[[0, 1, 2, 5], 'time'] = None
    exp = df.loc[[0, 3, 4, 6]].set_index('class')
    grouped = df.groupby('class')
    tm.assert_frame_equal(grouped.first(), exp)
    tm.assert_frame_equal(grouped.agg('first'), exp)
    tm.assert_frame_equal(grouped.agg({'time': 'first'}), exp)
    tm.assert_series_equal(grouped.time.first(), exp['time'])
    tm.assert_series_equal(grouped.time.agg('first'), exp['time'])
    exp = df.loc[[0, 3, 4, 7]].set_index('class')
    grouped = df.groupby('class')
    tm.assert_frame_equal(grouped.last(), exp)
    tm.assert_frame_equal(grouped.agg('last'), exp)
    tm.assert_frame_equal(grouped.agg({'time': 'last'}), exp)
    tm.assert_series_equal(grouped.time.last(), exp['time'])
    tm.assert_series_equal(grouped.time.agg('last'), exp['time'])
    exp = Series([2, 2, 2, 2], index=Index(list('ABCD'), name='class'), name='time')
    tm.assert_series_equal(grouped.time.agg(len), exp)
    tm.assert_series_equal(grouped.time.size(), exp)
    exp = Series([0, 1, 1, 2], index=Index(list('ABCD'), name='class'), name='time')
    tm.assert_series_equal(grouped.time.count(), exp)