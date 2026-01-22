from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_True(any_string_dtype):
    s = Series(['fooBAD__barBAD', np.nan, 'foo'], dtype=any_string_dtype)
    result = s.str.extract('.*(BAD[_]+).*(BAD)', expand=True)
    expected = DataFrame([['BAD__', 'BAD'], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)