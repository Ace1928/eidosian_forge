from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', ['first', 'last'])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(func: str, observed: bool):
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'), 'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)
    srs_grp = df.groupby(['a', 'b'], observed=observed)['c']
    result = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)