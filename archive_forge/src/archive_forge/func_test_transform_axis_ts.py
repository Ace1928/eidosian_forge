import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_axis_ts(tsframe):
    base = tsframe.iloc[0:5]
    r = len(base.index)
    c = len(base.columns)
    tso = DataFrame(np.random.default_rng(2).standard_normal((r, c)), index=base.index, columns=base.columns, dtype='float64')
    ts = tso
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform('mean')
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)
    ts = ts.T
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = ts.groupby(lambda x: x.weekday(), axis=1, group_keys=False)
    result = ts - grouped.transform('mean')
    expected = grouped.apply(lambda x: (x.T - x.mean(1)).T)
    tm.assert_frame_equal(result, expected)
    ts = tso.iloc[[1, 0] + list(range(2, len(base)))]
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform('mean')
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)
    ts = ts.T
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = ts.groupby(lambda x: x.weekday(), axis=1, group_keys=False)
    result = ts - grouped.transform('mean')
    expected = grouped.apply(lambda x: (x.T - x.mean(1)).T)
    tm.assert_frame_equal(result, expected)