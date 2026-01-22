from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_callable_named_groups(any_string_dtype):
    ser = Series(['Foo Bar Baz', np.nan], dtype=any_string_dtype)
    pat = '(?P<first>\\w+) (?P<middle>\\w+) (?P<last>\\w+)'
    repl = lambda m: m.group('middle').swapcase()
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.replace(pat, repl, regex=True)
    expected = Series(['bAR', np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)