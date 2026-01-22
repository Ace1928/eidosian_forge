from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('keys, expected_values, expected_index_levels', [('a', [15, 9, 0], CategoricalIndex([1, 2, 3], name='a')), (['a', 'b'], [7, 8, 0, 0, 0, 9, 0, 0, 0], [CategoricalIndex([1, 2, 3], name='a'), Index([4, 5, 6])]), (['a', 'a2'], [15, 0, 0, 0, 9, 0, 0, 0, 0], [CategoricalIndex([1, 2, 3], name='a'), CategoricalIndex([1, 2, 3], name='a')])])
@pytest.mark.parametrize('test_series', [True, False])
def test_unobserved_in_index(keys, expected_values, expected_index_levels, test_series):
    df = DataFrame({'a': Categorical([1, 1, 2], categories=[1, 2, 3]), 'a2': Categorical([1, 1, 2], categories=[1, 2, 3]), 'b': [4, 5, 6], 'c': [7, 8, 9]}).set_index(['a', 'a2'])
    if 'b' not in keys:
        df = df.drop(columns='b')
    gb = df.groupby(keys, observed=False)
    if test_series:
        gb = gb['c']
    result = gb.sum()
    if len(keys) == 1:
        index = expected_index_levels
    else:
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(expected_index_levels, codes=codes, names=keys)
    expected = DataFrame({'c': expected_values}, index=index)
    if test_series:
        expected = expected['c']
    tm.assert_equal(result, expected)