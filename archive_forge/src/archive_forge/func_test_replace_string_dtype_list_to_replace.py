import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_string_dtype_list_to_replace(self):
    ser = pd.Series(['abc', 'def'], dtype='string')
    res = ser.replace(['abc', 'any other string'], 'xyz')
    expected = pd.Series(['xyz', 'def'], dtype='string')
    tm.assert_series_equal(res, expected)