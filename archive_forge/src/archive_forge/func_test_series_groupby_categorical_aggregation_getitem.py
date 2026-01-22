from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_series_groupby_categorical_aggregation_getitem():
    d = {'foo': [10, 8, 4, 1], 'bar': [10, 20, 30, 40], 'baz': ['d', 'c', 'd', 'c']}
    df = DataFrame(d)
    cat = pd.cut(df['foo'], np.linspace(0, 20, 5))
    df['range'] = cat
    groups = df.groupby(['range', 'baz'], as_index=True, sort=True, observed=False)
    result = groups['foo'].agg('mean')
    expected = groups.agg('mean')['foo']
    tm.assert_series_equal(result, expected)