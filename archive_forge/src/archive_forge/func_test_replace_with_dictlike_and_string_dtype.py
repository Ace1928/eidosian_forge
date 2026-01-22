import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype):
    ser = pd.Series(['one', 'two', np.nan], dtype=nullable_string_dtype)
    expected = pd.Series(['1', '2', np.nan], dtype=nullable_string_dtype)
    result = ser.replace({'one': '1', 'two': '2'})
    tm.assert_series_equal(expected, result)