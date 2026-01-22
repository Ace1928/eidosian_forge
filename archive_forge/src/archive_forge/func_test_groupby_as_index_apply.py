from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_as_index_apply():
    df = DataFrame({'item_id': ['b', 'b', 'a', 'c', 'a', 'b'], 'user_id': [1, 2, 1, 1, 3, 1], 'time': range(6)})
    g_as = df.groupby('user_id', as_index=True)
    g_not_as = df.groupby('user_id', as_index=False)
    res_as = g_as.head(2).index
    res_not_as = g_not_as.head(2).index
    exp = Index([0, 1, 2, 4])
    tm.assert_index_equal(res_as, exp)
    tm.assert_index_equal(res_not_as, exp)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res_as_apply = g_as.apply(lambda x: x.head(2)).index
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res_not_as_apply = g_not_as.apply(lambda x: x.head(2)).index
    exp_not_as_apply = MultiIndex.from_tuples([(0, 0), (0, 2), (1, 1), (2, 4)])
    tp = [(1, 0), (1, 2), (2, 1), (3, 4)]
    exp_as_apply = MultiIndex.from_tuples(tp, names=['user_id', None])
    tm.assert_index_equal(res_as_apply, exp_as_apply)
    tm.assert_index_equal(res_not_as_apply, exp_not_as_apply)
    ind = Index(list('abcde'))
    df = DataFrame([[1, 2], [2, 3], [1, 4], [1, 5], [2, 6]], index=ind)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res = df.groupby(0, as_index=False, group_keys=False).apply(lambda x: x).index
    tm.assert_index_equal(res, ind)