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
def test_37692(indexer_al):
    ser = Series([1, 2, 3], index=['a', 'b', 'c'])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        indexer_al(ser)['b'] = 'test'
    expected = Series([1, 'test', 3], index=['a', 'b', 'c'], dtype=object)
    tm.assert_series_equal(ser, expected)