from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_contains_na_kwarg_for_object_category():
    values = Series(['a', 'b', 'c', 'a', np.nan], dtype='category')
    result = values.str.contains('a', na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)
    result = values.str.contains('a', na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)
    values = Series(['a', 'b', 'c', 'a', np.nan])
    result = values.str.contains('a', na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)
    result = values.str.contains('a', na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)