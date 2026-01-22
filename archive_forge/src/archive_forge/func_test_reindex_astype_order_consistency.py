import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_astype_order_consistency():
    ser = Series([1, 2, 3], index=[2, 0, 1])
    new_index = [0, 1, 2]
    temp_dtype = 'category'
    new_dtype = str
    result = ser.reindex(new_index).astype(temp_dtype).astype(new_dtype)
    expected = ser.astype(temp_dtype).reindex(new_index).astype(new_dtype)
    tm.assert_series_equal(result, expected)