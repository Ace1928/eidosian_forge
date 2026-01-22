import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_take_slice_raises():
    ser = Series([-1, 5, 6, 2, 4])
    msg = 'Series.take requires a sequence of integers, not slice'
    with pytest.raises(TypeError, match=msg):
        ser.take(slice(0, 3, 1))