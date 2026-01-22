from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('regex', [True, False])
def test_replace_regex_single_character(regex, any_string_dtype):
    s = Series(['a.b', '.', 'b', np.nan, ''], dtype=any_string_dtype)
    result = s.str.replace('.', 'a', regex=regex)
    if regex:
        expected = Series(['aaa', 'a', 'a', np.nan, ''], dtype=any_string_dtype)
    else:
        expected = Series(['aab', 'a', 'b', np.nan, ''], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)