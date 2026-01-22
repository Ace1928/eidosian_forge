from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_getitem_list_of_columns(self):
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8), 'E': np.random.default_rng(2).standard_normal(8)})
    result = df.groupby('A')[['C', 'D']].mean()
    result2 = df.groupby('A')[df.columns[2:4]].mean()
    expected = df.loc[:, ['A', 'C', 'D']].groupby('A').mean()
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)