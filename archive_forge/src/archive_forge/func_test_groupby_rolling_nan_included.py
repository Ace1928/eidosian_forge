from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_groupby_rolling_nan_included():
    data = {'group': ['g1', np.nan, 'g1', 'g2', np.nan], 'B': [0, 1, 2, 3, 4]}
    df = DataFrame(data)
    result = df.groupby('group', dropna=False).rolling(1, min_periods=1).mean()
    expected = DataFrame({'B': [0.0, 2.0, 3.0, 1.0, 4.0]}, index=MultiIndex([['g1', 'g2', np.nan], [0, 1, 2, 3, 4]], [[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]], names=['group', None]))
    tm.assert_frame_equal(result, expected)