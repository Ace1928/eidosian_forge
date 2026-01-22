from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_True_single_capture_group(index_or_series, any_string_dtype):
    s_or_idx = index_or_series(['A1', 'A2'], dtype=any_string_dtype)
    result = s_or_idx.str.extract('(?P<uno>A)\\d', expand=True)
    expected = DataFrame({'uno': ['A', 'A']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)