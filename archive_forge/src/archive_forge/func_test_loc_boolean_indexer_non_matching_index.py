from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_loc_boolean_indexer_non_matching_index():
    ser = Series([1])
    result = ser.loc[Series([NA, False], dtype='boolean')]
    expected = Series([], dtype='int64')
    tm.assert_series_equal(result, expected)