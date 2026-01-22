import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype):
    values = ['a', '1']
    ser = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(ser, errors='coerce')
    expected = Series([pd.NA, 1], dtype='Int64')
    tm.assert_series_equal(result, expected)