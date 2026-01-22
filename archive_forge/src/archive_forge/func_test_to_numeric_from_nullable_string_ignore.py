import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_numeric_from_nullable_string_ignore(nullable_string_dtype):
    values = ['a', '1']
    ser = Series(values, dtype=nullable_string_dtype)
    expected = ser.copy()
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_numeric(ser, errors='ignore')
    tm.assert_series_equal(result, expected)