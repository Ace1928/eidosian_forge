from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_single_series_name_is_preserved(any_string_dtype):
    s = Series(['a3', 'b3', 'c2'], name='bob', dtype=any_string_dtype)
    result = s.str.extract('(?P<sue>[a-z])', expand=False)
    expected = Series(['a', 'b', 'c'], name='sue', dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)