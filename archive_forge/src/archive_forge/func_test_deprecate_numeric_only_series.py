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
@pytest.mark.parametrize('dtype', [bool, int, float, object])
def test_deprecate_numeric_only_series(dtype, groupby_func, request):
    if groupby_func == 'corrwith':
        msg = 'corrwith is not implemented on SeriesGroupBy'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    grouper = [0, 0, 1]
    ser = Series([1, 0, 0], dtype=dtype)
    gb = ser.groupby(grouper)
    method = getattr(gb, groupby_func)
    expected_ser = Series([1, 0, 0])
    expected_gb = expected_ser.groupby(grouper)
    expected_method = getattr(expected_gb, groupby_func)
    args = get_groupby_method_args(groupby_func, ser)
    fails_on_numeric_object = ('corr', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'idxmax', 'idxmin', 'quantile')
    obj_result = ('first', 'last', 'nth', 'bfill', 'ffill', 'shift', 'sum', 'diff', 'pct_change', 'var', 'mean', 'median', 'min', 'max', 'prod')
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func in ('idxmax', 'idxmin'):
            msg = 'not allowed for this dtype'
        elif groupby_func == 'quantile':
            msg = "cannot be performed against 'object' dtypes"
        else:
            msg = 'is not supported for object dtype'
        with pytest.raises(TypeError, match=msg):
            method(*args)
    elif dtype is object:
        result = method(*args)
        expected = expected_method(*args)
        if groupby_func in obj_result:
            expected = expected.astype(object)
        tm.assert_series_equal(result, expected)
    has_numeric_only = ('first', 'last', 'max', 'mean', 'median', 'min', 'prod', 'quantile', 'sem', 'skew', 'std', 'sum', 'var', 'cummax', 'cummin', 'cumprod', 'cumsum')
    if groupby_func not in has_numeric_only:
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype is object:
        msg = '|'.join(['SeriesGroupBy.sem called with numeric_only=True and dtype object', 'Series.skew does not allow numeric_only=True with non-numeric', 'cum(sum|prod|min|max) is not supported for object dtype', 'Cannot use numeric_only=True with SeriesGroupBy\\..* and non-numeric'])
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    else:
        result = method(*args, numeric_only=True)
        expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)