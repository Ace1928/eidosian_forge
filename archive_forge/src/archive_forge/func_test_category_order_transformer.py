from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index_kind', ['single', 'multi'])
@pytest.mark.parametrize('ordered', [True, False])
def test_category_order_transformer(as_index, sort, observed, transformation_func, index_kind, ordered):
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    args = get_groupby_method_args(transformation_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    warn = FutureWarning if transformation_func == 'fillna' else None
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        op_result = getattr(gb, transformation_func)(*args)
    result = op_result.index.get_level_values('a').categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)