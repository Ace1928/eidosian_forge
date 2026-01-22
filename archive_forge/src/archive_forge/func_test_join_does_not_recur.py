import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_join_does_not_recur(self):
    df = DataFrame(np.ones((3, 2)), index=date_range('2020-01-01', periods=3), columns=period_range('2020-01-01', periods=2))
    ser = df.iloc[:2, 0]
    res = ser.index.join(df.columns, how='outer')
    expected = Index([ser.index[0], ser.index[1], df.columns[0], df.columns[1]], object)
    tm.assert_index_equal(res, expected)