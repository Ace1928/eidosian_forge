from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_count_with_min_periods(frame_or_series):
    result = frame_or_series(range(5)).rolling(3, min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 3.0, 3.0])
    tm.assert_equal(result, expected)