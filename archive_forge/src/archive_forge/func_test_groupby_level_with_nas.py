from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('sort', [True, False])
def test_groupby_level_with_nas(self, sort):
    index = MultiIndex(levels=[[1, 0], [0, 1, 2, 3]], codes=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]])
    s = Series(np.arange(8.0), index=index)
    result = s.groupby(level=0, sort=sort).sum()
    expected = Series([6.0, 22.0], index=[0, 1])
    tm.assert_series_equal(result, expected)
    index = MultiIndex(levels=[[1, 0], [0, 1, 2, 3]], codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]])
    s = Series(np.arange(8.0), index=index)
    result = s.groupby(level=0, sort=sort).sum()
    expected = Series([6.0, 18.0], index=[0.0, 1.0])
    tm.assert_series_equal(result, expected)