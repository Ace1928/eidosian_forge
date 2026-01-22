import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_sequence_mismatched_length_raises(self, data, as_array):
    ser = pd.Series(data)
    original = ser.copy()
    value = [data[0]]
    if as_array:
        value = data._from_sequence(value, dtype=data.dtype)
    xpr = 'cannot set using a {} indexer with a different length'
    with pytest.raises(ValueError, match=xpr.format('list-like')):
        ser[[0, 1]] = value
    tm.assert_series_equal(ser, original)
    with pytest.raises(ValueError, match=xpr.format('slice')):
        ser[slice(3)] = value
    tm.assert_series_equal(ser, original)