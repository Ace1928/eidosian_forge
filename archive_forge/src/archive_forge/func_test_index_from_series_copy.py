import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_index_from_series_copy(using_copy_on_write):
    ser = Series([1, 2])
    idx = Index(ser, copy=True)
    arr = get_array(ser)
    ser.iloc[0] = 100
    assert np.shares_memory(get_array(ser), arr)