import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_transformation_func(transformation_func):
    df = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'baz'], 'B': [1, 2, np.nan, 3, 3, np.nan, 4]}, index=date_range('2020-01-01', '2020-01-07'))
    if transformation_func == 'cumcount':
        test_op = lambda x: x.transform('cumcount')
        mock_op = lambda x: Series(range(len(x)), x.index)
    elif transformation_func == 'fillna':
        test_op = lambda x: x.transform('fillna', value=0)
        mock_op = lambda x: x.fillna(value=0)
    elif transformation_func == 'ngroup':
        test_op = lambda x: x.transform('ngroup')
        counter = -1

        def mock_op(x):
            nonlocal counter
            counter += 1
            return Series(counter, index=x.index)
    else:
        test_op = lambda x: x.transform(transformation_func)
        mock_op = lambda x: getattr(x, transformation_func)()
    if transformation_func == 'pct_change':
        msg = "The default fill_method='pad' in DataFrame.pct_change is deprecated"
        groupby_msg = "The default fill_method='ffill' in DataFrameGroupBy.pct_change is deprecated"
        warn = FutureWarning
        groupby_warn = FutureWarning
    elif transformation_func == 'fillna':
        msg = ''
        groupby_msg = 'DataFrameGroupBy.fillna is deprecated'
        warn = None
        groupby_warn = FutureWarning
    else:
        msg = groupby_msg = ''
        warn = groupby_warn = None
    with tm.assert_produces_warning(groupby_warn, match=groupby_msg):
        result = test_op(df.groupby('A'))
    groups = [df[['B']].iloc[4:6], df[['B']].iloc[6:], df[['B']].iloc[:4]]
    with tm.assert_produces_warning(warn, match=msg):
        expected = concat([mock_op(g) for g in groups]).sort_index()
    expected = expected.set_axis(df.index)
    if transformation_func in ('cumcount', 'ngroup'):
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)