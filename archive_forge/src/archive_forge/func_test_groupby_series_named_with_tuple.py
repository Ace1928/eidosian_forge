from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('index', [[0, 1, 2, 3], ['a', 'b', 'c', 'd'], [Timestamp(2021, 7, 28 + i) for i in range(4)]])
def test_groupby_series_named_with_tuple(self, frame_or_series, index):
    obj = frame_or_series([1, 2, 3, 4], index=index)
    groups = Series([1, 0, 1, 0], index=index, name=('a', 'a'))
    result = obj.groupby(groups).last()
    expected = frame_or_series([4, 3])
    expected.index.name = ('a', 'a')
    tm.assert_equal(result, expected)