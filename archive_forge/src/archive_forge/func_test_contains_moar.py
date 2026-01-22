from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_contains_moar(any_string_dtype):
    s = Series(['A', 'B', 'C', 'Aaba', 'Baca', '', np.nan, 'CABA', 'dog', 'cat'], dtype=any_string_dtype)
    result = s.str.contains('a')
    expected_dtype = 'object' if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([False, False, False, True, True, False, np.nan, False, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.contains('a', case=False)
    expected = Series([True, False, False, True, True, False, np.nan, True, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.contains('Aa')
    expected = Series([False, False, False, True, False, False, np.nan, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.contains('ba')
    expected = Series([False, False, False, True, False, False, np.nan, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.contains('ba', case=False)
    expected = Series([False, False, False, True, True, False, np.nan, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)