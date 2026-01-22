import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('other', [Interval(0, 1, closed='right'), IntervalArray.from_breaks([1, 2, 3, 4], closed='right')])
def test_where_raises(self, other):
    ser = pd.Series(IntervalArray.from_breaks([1, 2, 3, 4], closed='left'))
    mask = np.array([True, False, True])
    match = "'value.closed' is 'right', expected 'left'."
    with pytest.raises(ValueError, match=match):
        ser.array._where(mask, other)
    res = ser.where(mask, other=other)
    expected = ser.astype(object).where(mask, other)
    tm.assert_series_equal(res, expected)