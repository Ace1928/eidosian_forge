from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('as_index, expected', [[False, DataFrame([[1, 1, 1], [2, 2, 1]], columns=Index(['a', 'b', None], dtype=object))], [True, Series([1, 1], index=MultiIndex.from_tuples([(1, 1), (2, 2)], names=['a', 'b']))]])
def test_apply_as_index_constant_lambda(as_index, expected):
    df = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 1, 2, 2], 'c': [1, 1, 1, 1]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(['a', 'b'], as_index=as_index).apply(lambda x: 1)
    tm.assert_equal(result, expected)