import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_compare():
    midx = MultiIndex.from_product([[0, 1]])
    expected = Series([True, True])
    result = Series(midx == midx)
    tm.assert_series_equal(result, expected)
    expected = Series([False, False])
    result = Series(midx > midx)
    tm.assert_series_equal(result, expected)