from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_even_number_window_alignment():
    s = Series(range(3), index=date_range(start='2020-01-01', freq='D', periods=3))
    result = s.rolling(window='2D', min_periods=1, center=True).mean()
    expected = Series([0.5, 1.5, 2], index=s.index)
    tm.assert_series_equal(result, expected)