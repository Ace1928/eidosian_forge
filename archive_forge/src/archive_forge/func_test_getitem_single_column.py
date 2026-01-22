from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_getitem_single_column(self):
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8), 'E': np.random.default_rng(2).standard_normal(8)})
    result = df.groupby('A')['C'].mean()
    as_frame = df.loc[:, ['A', 'C']].groupby('A').mean()
    as_series = as_frame.iloc[:, 0]
    expected = as_series
    tm.assert_series_equal(result, expected)