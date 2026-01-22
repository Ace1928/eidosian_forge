from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, exp', [['partition', [('abc', '', ''), ('cde', '', ''), np.nan, ('fgh', '', ''), None]], ['rpartition', [('', '', 'abc'), ('', '', 'cde'), np.nan, ('', '', 'fgh'), None]]])
def test_partition_series_not_split(any_string_dtype, method, exp):
    s = Series(['abc', 'cde', np.nan, 'fgh', None], dtype=any_string_dtype)
    result = getattr(s.str, method)('_', expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)