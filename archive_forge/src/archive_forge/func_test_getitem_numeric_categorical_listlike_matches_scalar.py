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
def test_getitem_numeric_categorical_listlike_matches_scalar(self):
    ser = Series(['a', 'b', 'c'], index=pd.CategoricalIndex([2, 1, 0]))
    assert ser[0] == 'c'
    res = ser[[0]]
    expected = ser.iloc[-1:]
    tm.assert_series_equal(res, expected)
    res2 = ser[[0, 1, 2]]
    tm.assert_series_equal(res2, ser.iloc[::-1])