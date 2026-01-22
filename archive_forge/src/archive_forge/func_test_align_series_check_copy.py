from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_series_check_copy(self):
    df = DataFrame({0: [1, 2]})
    ser = Series([1], name=0)
    expected = ser.copy()
    result, other = df.align(ser, axis=1)
    ser.iloc[0] = 100
    tm.assert_series_equal(other, expected)