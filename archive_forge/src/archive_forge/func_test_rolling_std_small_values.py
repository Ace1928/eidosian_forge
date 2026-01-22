from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_std_small_values():
    s = Series([5.4e-07, 5.3e-07, 5.4e-07])
    result = s.rolling(2).std()
    expected = Series([np.nan, 7.071068e-09, 7.071068e-09])
    tm.assert_series_equal(result, expected, atol=1e-15, rtol=1e-15)