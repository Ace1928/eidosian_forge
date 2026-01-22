from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_creation_bug(self):
    df = DataFrame({'A': [0, 0, 1, 1, 2, 2], 'B': [1, 2, 3, 4, 5, 6]})
    g = df.groupby('A')
    expected = g.sum()
    g = df.groupby(Grouper(key='A'))
    result = g.sum()
    tm.assert_frame_equal(result, expected)
    msg = 'Grouper axis keyword is deprecated and will be removed'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gpr = Grouper(key='A', axis=0)
    g = df.groupby(gpr)
    result = g.sum()
    tm.assert_frame_equal(result, expected)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = g.apply(lambda x: x.sum())
    expected['A'] = [0, 2, 4]
    expected = expected.loc[:, ['A', 'B']]
    tm.assert_frame_equal(result, expected)