from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_optional_groups(any_string_dtype):
    s = Series(['A11', 'B22', 'C33'], dtype=any_string_dtype)
    result = s.str.extract('([AB])([123])(?:[123])', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], [np.nan, np.nan]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    s = Series(['A1', 'B2', '3'], dtype=any_string_dtype)
    result = s.str.extract('(?P<letter>[AB])?(?P<number>[123])', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], [np.nan, '3']], columns=['letter', 'number'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    s = Series(['A1', 'B2', 'C'], dtype=any_string_dtype)
    result = s.str.extract('(?P<letter>[ABC])(?P<number>[123])?', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], ['C', np.nan]], columns=['letter', 'number'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)