import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_dispatch_transform(tsframe):
    df = tsframe[::5].reindex(tsframe.index)
    grouped = df.groupby(lambda x: x.month)
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        filled = grouped.fillna(method='pad')
    msg = "Series.fillna with 'method' is deprecated"
    fillit = lambda x: x.fillna(method='pad')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(lambda x: x.month).transform(fillit)
    tm.assert_frame_equal(filled, expected)