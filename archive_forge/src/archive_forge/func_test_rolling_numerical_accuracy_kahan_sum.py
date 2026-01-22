from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_numerical_accuracy_kahan_sum():
    df = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=['x'])
    result = df['x'].rolling(3).sum()
    expected = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name='x')
    tm.assert_series_equal(result, expected)