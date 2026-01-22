from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_fullmatch_dollar_literal(any_string_dtype):
    ser = Series(['foo', 'foo$foo', np.nan, 'foo$'], dtype=any_string_dtype)
    result = ser.str.fullmatch('foo\\$')
    expected_dtype = 'object' if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([False, False, np.nan, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)