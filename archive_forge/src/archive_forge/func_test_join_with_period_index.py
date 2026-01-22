from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_join_with_period_index(self, join_type):
    df = DataFrame(np.ones((10, 2)), index=date_range('2020-01-01', periods=10), columns=period_range('2020-01-01', periods=2))
    s = df.iloc[:5, 0]
    expected = df.columns.astype('O').join(s.index, how=join_type)
    result = df.columns.join(s.index, how=join_type)
    tm.assert_index_equal(expected, result)