import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_cummin(dtypes_for_minmax):
    dtype = dtypes_for_minmax[0]
    min_val = dtypes_for_minmax[1]
    base_df = DataFrame({'A': [1, 1, 1, 1, 2, 2, 2, 2], 'B': [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]
    df = base_df.astype(dtype)
    expected = DataFrame({'B': expected_mins}).astype(dtype)
    result = df.groupby('A').cummin()
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A', group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected)
    df.loc[[2, 6], 'B'] = min_val
    df.loc[[1, 5], 'B'] = min_val + 1
    expected.loc[[2, 3, 6, 7], 'B'] = min_val
    expected.loc[[1, 5], 'B'] = min_val + 1
    result = df.groupby('A').cummin()
    tm.assert_frame_equal(result, expected, check_exact=True)
    expected = df.groupby('A', group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected, check_exact=True)
    base_df = base_df.astype({'B': 'float'})
    base_df.loc[[0, 2, 4, 6], 'B'] = np.nan
    expected = DataFrame({'B': [np.nan, 4, np.nan, 2, np.nan, 3, np.nan, 1]})
    result = base_df.groupby('A').cummin()
    tm.assert_frame_equal(result, expected)
    expected = base_df.groupby('A', group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': [1], 'b': pd.to_datetime(['2001'])})
    expected = Series(pd.to_datetime('2001'), index=[0], name='b')
    result = df.groupby('a')['b'].cummin()
    tm.assert_series_equal(expected, result)
    df = DataFrame({'a': [1, 2, 1], 'b': [1, 2, 2]})
    result = df.groupby('a').b.cummin()
    expected = Series([1, 2, 1], name='b')
    tm.assert_series_equal(result, expected)