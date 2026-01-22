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
def test_groupby_sort_multi():
    df = DataFrame({'a': ['foo', 'bar', 'baz'], 'b': [3, 2, 1], 'c': [0, 1, 2], 'd': np.random.default_rng(2).standard_normal(3)})
    tups = [tuple(row) for row in df[['a', 'b', 'c']].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(['a', 'b', 'c'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])
    tups = [tuple(row) for row in df[['c', 'a', 'b']].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(['c', 'a', 'b'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups)
    tups = [tuple(x) for x in df[['b', 'c', 'a']].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(['b', 'c', 'a'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])
    df = DataFrame({'a': [0, 1, 2, 0, 1, 2], 'b': [0, 0, 0, 1, 1, 1], 'd': np.random.default_rng(2).standard_normal(6)})
    grouped = df.groupby(['a', 'b'])['d']
    result = grouped.sum()

    def _check_groupby(df, result, keys, field, f=lambda x: x.sum()):
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        expected = f(df.groupby(tups)[field])
        for k, v in expected.items():
            assert result[k] == v
    _check_groupby(df, result, ['a', 'b'], 'd')