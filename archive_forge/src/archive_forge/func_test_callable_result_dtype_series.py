import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('keys, agg_index', [(['a'], Index([1], name='a')), (['a', 'b'], MultiIndex([[1], [2]], [[0], [0]], names=['a', 'b']))])
@pytest.mark.parametrize('input', [True, 1, 1.0])
@pytest.mark.parametrize('dtype', [bool, int, float])
@pytest.mark.parametrize('method', ['apply', 'aggregate', 'transform'])
def test_callable_result_dtype_series(keys, agg_index, input, dtype, method):
    df = DataFrame({'a': [1], 'b': [2], 'c': [input]})
    op = getattr(df.groupby(keys)['c'], method)
    result = op(lambda x: x.astype(dtype).iloc[0])
    expected_index = pd.RangeIndex(0, 1) if method == 'transform' else agg_index
    expected = Series([df['c'].iloc[0]], index=expected_index, name='c').astype(dtype)
    tm.assert_series_equal(result, expected)