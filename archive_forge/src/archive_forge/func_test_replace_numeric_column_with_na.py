import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('val', [0, 0.5])
def test_replace_numeric_column_with_na(self, val):
    ser = pd.Series([val, 1])
    expected = pd.Series([val, pd.NA])
    result = ser.replace(to_replace=1, value=pd.NA)
    tm.assert_series_equal(result, expected)
    ser.replace(to_replace=1, value=pd.NA, inplace=True)
    tm.assert_series_equal(ser, expected)