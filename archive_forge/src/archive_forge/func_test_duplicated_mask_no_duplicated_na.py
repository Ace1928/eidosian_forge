import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_duplicated_mask_no_duplicated_na(keep):
    ser = Series([1, 2, NA], dtype='Int64')
    result = ser.duplicated(keep=keep)
    expected = Series([False, False, False])
    tm.assert_series_equal(result, expected)