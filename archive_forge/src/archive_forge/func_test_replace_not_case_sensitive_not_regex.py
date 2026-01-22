from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_not_case_sensitive_not_regex(any_string_dtype):
    ser = Series(['A.', 'a.', 'Ab', 'ab', np.nan], dtype=any_string_dtype)
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.replace('a', 'c', case=False, regex=False)
    expected = Series(['c.', 'c.', 'cb', 'cb', np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.replace('a.', 'c.', case=False, regex=False)
    expected = Series(['c.', 'c.', 'Ab', 'ab', np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)