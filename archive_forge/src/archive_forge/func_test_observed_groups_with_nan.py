from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_observed_groups_with_nan(observed):
    df = DataFrame({'cat': Categorical(['a', np.nan, 'a'], categories=['a', 'b', 'd']), 'vals': [1, 2, 3]})
    g = df.groupby('cat', observed=observed)
    result = g.groups
    if observed:
        expected = {'a': Index([0, 2], dtype='int64')}
    else:
        expected = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype='int64'), 'd': Index([], dtype='int64')}
    tm.assert_dict_equal(result, expected)