from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_to_series_name():
    df = DataFrame.from_dict({'a': ['a', 'b', 'a', 'b'], 'b1': ['aa', 'ac', 'ac', 'ad'], 'b2': ['aa', 'aa', 'aa', 'ac']})
    grp = df.groupby('a')[['b1', 'b2']]
    result = grp.apply(lambda x: x.unstack().value_counts())
    expected_idx = MultiIndex.from_arrays(arrays=[['a', 'a', 'b', 'b', 'b'], ['aa', 'ac', 'ac', 'ad', 'aa']], names=['a', None])
    expected = Series([3, 1, 2, 1, 1], index=expected_idx, name='count')
    tm.assert_series_equal(result, expected)