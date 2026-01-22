from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_group_on_empty_multiindex(transformation_func, request):
    df = DataFrame(data=[[1, Timestamp('today'), 3, 4]], columns=['col_1', 'col_2', 'col_3', 'col_4'])
    df['col_3'] = df['col_3'].astype(int)
    df['col_4'] = df['col_4'].astype(int)
    df = df.set_index(['col_1', 'col_2'])
    if transformation_func == 'fillna':
        args = ('ffill',)
    else:
        args = ()
    warn = FutureWarning if transformation_func == 'fillna' else None
    warn_msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = df.iloc[:0].groupby(['col_1']).transform(transformation_func, *args)
    with tm.assert_produces_warning(warn, match=warn_msg):
        expected = df.groupby(['col_1']).transform(transformation_func, *args).iloc[:0]
    if transformation_func in ('diff', 'shift'):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)
    warn_msg = 'SeriesGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = df['col_3'].iloc[:0].groupby(['col_1']).transform(transformation_func, *args)
    warn_msg = 'SeriesGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=warn_msg):
        expected = df['col_3'].groupby(['col_1']).transform(transformation_func, *args).iloc[:0]
    if transformation_func in ('diff', 'shift'):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)