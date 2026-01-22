from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('as_index, expected', [(True, Series(index=MultiIndex.from_arrays([Series([1, 1, 2], dtype='category'), [1, 2, 2]], names=['a', 'b']), data=[1, 2, 3], name='x')), (False, DataFrame({'a': Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2], 'x': [1, 2, 3]}))])
def test_groupby_agg_observed_true_single_column(as_index, expected):
    df = DataFrame({'a': Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2], 'x': [1, 2, 3]})
    result = df.groupby(['a', 'b'], as_index=as_index, observed=True)['x'].sum()
    tm.assert_equal(result, expected)