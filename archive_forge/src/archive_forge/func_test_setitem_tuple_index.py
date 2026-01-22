import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_tuple_index(self, data):
    ser = pd.Series(data[:2], index=[(0, 0), (0, 1)])
    expected = pd.Series(data.take([1, 1]), index=ser.index)
    ser[0, 0] = data[1]
    tm.assert_series_equal(ser, expected)