from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_single_group_returns_frame(any_string_dtype):
    s = Series(['a3', 'b3', 'c2'], name='series_name', dtype=any_string_dtype)
    result = s.str.extract('(?P<letter>[a-z])', expand=True)
    expected = DataFrame({'letter': ['a', 'b', 'c']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)