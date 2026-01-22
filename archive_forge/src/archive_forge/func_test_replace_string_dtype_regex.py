import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_string_dtype_regex(self):
    ser = pd.Series(['A', 'B'], dtype='string')
    res = ser.replace('.', 'C', regex=True)
    expected = pd.Series(['C', 'C'], dtype='string')
    tm.assert_series_equal(res, expected)