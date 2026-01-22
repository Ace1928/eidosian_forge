from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_match_na_kwarg(any_string_dtype):
    s = Series(['a', 'b', np.nan], dtype=any_string_dtype)
    result = s.str.match('a', na=False)
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.match('a')
    expected_dtype = 'object' if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([True, False, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)