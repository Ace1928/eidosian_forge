from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_regex(any_string_dtype):
    s = Series(['a', 'b', 'ac', np.nan, ''], dtype=any_string_dtype)
    result = s.str.replace('^.$', 'a', regex=True)
    expected = Series(['a', 'a', 'ac', np.nan, ''], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)