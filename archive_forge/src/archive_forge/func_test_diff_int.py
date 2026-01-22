import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_diff_int(self):
    a = 10000000000000000
    b = a + 1
    ser = Series([a, b])
    result = ser.diff()
    assert result[1] == 1