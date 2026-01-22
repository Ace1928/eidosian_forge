import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('d', [4, 'd'])
def test_empty_frame_groupby_dtypes_consistency(self, d):
    group_keys = ['a', 'b', 'c']
    df = DataFrame({'a': [1], 'b': [2], 'c': [3], 'd': [d]})
    g = df[df.a == 2].groupby(group_keys)
    result = g.first().index
    expected = MultiIndex(levels=[[1], [2], [3]], codes=[[], [], []], names=['a', 'b', 'c'])
    tm.assert_index_equal(result, expected)