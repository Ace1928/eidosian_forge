from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_setitem_with_bool_indexer():
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.pop('b').copy()
    result[[True, False, False]] = 9
    expected = Series(data=[9, 5, 6], name='b')
    tm.assert_series_equal(result, expected)
    df.loc[[True, False, False], 'a'] = 10
    expected = DataFrame({'a': [10, 2, 3]})
    tm.assert_frame_equal(df, expected)