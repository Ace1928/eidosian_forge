from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_empty_window_median_quantile():
    expected = Series([np.nan, np.nan, np.nan])
    roll = Series(np.arange(3)).rolling(0)
    result = roll.median()
    tm.assert_series_equal(result, expected)
    result = roll.quantile(0.1)
    tm.assert_series_equal(result, expected)