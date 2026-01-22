import pandas as pd
import pandas._testing as tm
def test_array_setitem_nullable_boolean_mask():
    ser = pd.Series([1, 2], dtype='Int64')
    result = ser.where(ser > 1)
    expected = pd.Series([pd.NA, 2], dtype='Int64')
    tm.assert_series_equal(result, expected)