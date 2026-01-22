from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_compiled_regex_callable(any_string_dtype):
    ser = Series(['fooBAD__barBAD', np.nan], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()
    pat = re.compile('[a-z][A-Z]{2}')
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.replace(pat, repl, n=2, regex=True)
    expected = Series(['foObaD__baRbaD', np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)