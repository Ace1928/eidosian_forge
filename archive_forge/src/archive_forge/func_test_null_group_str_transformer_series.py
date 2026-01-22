import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_null_group_str_transformer_series(dropna, transformation_func):
    ser = Series([1, 2, 2], index=[1, 2, 3])
    args = get_groupby_method_args(transformation_func, ser)
    gb = ser.groupby([1, 1, np.nan], dropna=dropna)
    buffer = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == 'cumcount':
            res = Series(range(len(group)), index=group.index)
        elif transformation_func == 'ngroup':
            res = Series(k, index=group.index)
        else:
            res = getattr(group, transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype = object if transformation_func in ('any', 'all') else None
        buffer.append(Series([np.nan], index=[3], dtype=dtype))
    expected = concat(buffer)
    warn = FutureWarning if transformation_func == 'fillna' else None
    msg = 'SeriesGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)