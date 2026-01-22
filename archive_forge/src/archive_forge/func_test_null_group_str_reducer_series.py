import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_null_group_str_reducer_series(request, dropna, reduction_func):
    index = [1, 2, 3, 4]
    ser = Series([1, 2, 2, 3], index=index)
    gb = ser.groupby([1, 1, np.nan, np.nan], dropna=dropna)
    if reduction_func == 'corrwith':
        assert not hasattr(gb, reduction_func)
        return
    args = get_groupby_method_args(reduction_func, ser)
    if reduction_func == 'first':
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == 'last':
        expected = Series([2, 2, 3, 3], index=index)
    elif reduction_func == 'nth':
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == 'size':
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == 'corrwith':
        expected = Series([1, 1, 2, 2], index=index)
    else:
        expected_gb = ser.groupby([1, 1, np.nan, np.nan], dropna=False)
        buffer = []
        for idx, group in expected_gb:
            res = getattr(group, reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer)
    if dropna:
        dtype = object if reduction_func in ('any', 'all') else float
        expected = expected.astype(dtype)
        expected.iloc[[2, 3]] = np.nan
    result = gb.transform(reduction_func, *args)
    tm.assert_series_equal(result, expected)