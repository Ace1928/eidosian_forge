from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_lower_upper(any_string_dtype):
    s = Series(['om', np.nan, 'nom', 'nom'], dtype=any_string_dtype)
    result = s.str.upper()
    expected = Series(['OM', np.nan, 'NOM', 'NOM'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = result.str.lower()
    tm.assert_series_equal(result, s)