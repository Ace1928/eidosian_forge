from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_partial_str_slice_with_datetimeindex(self):
    arr = date_range('1/1/2008', '1/1/2009')
    ser = arr.to_series()
    result = ser['2008']
    rng = date_range(start='2008-01-01', end='2008-12-31')
    expected = Series(rng, index=rng)
    tm.assert_series_equal(result, expected)