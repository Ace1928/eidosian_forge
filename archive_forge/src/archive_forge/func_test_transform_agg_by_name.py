import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_agg_by_name(request, reduction_func, frame_or_series):
    func = reduction_func
    obj = DataFrame({'a': [0, 0, 0, 1, 1, 1], 'b': range(6)}, index=['A', 'B', 'C', 'D', 'E', 'F'])
    if frame_or_series is Series:
        obj = obj['a']
    g = obj.groupby(np.repeat([0, 1], 3))
    if func == 'corrwith' and isinstance(obj, Series):
        assert not hasattr(g, func)
        return
    args = get_groupby_method_args(reduction_func, obj)
    result = g.transform(func, *args)
    tm.assert_index_equal(result.index, obj.index)
    if func not in ('ngroup', 'size') and obj.ndim == 2:
        tm.assert_index_equal(result.columns, obj.columns)
    assert len(set(DataFrame(result).iloc[-3:, -1])) == 1