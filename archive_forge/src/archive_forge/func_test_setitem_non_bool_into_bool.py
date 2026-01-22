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
@pytest.mark.parametrize('unique', [True, False])
@pytest.mark.parametrize('val', [3, 3.0, '3'], ids=type)
def test_setitem_non_bool_into_bool(self, val, indexer_sli, unique):
    ser = Series([True, False])
    if not unique:
        ser.index = [1, 1]
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        indexer_sli(ser)[1] = val
    assert type(ser.iloc[1]) == type(val)
    expected = Series([True, val], dtype=object, index=ser.index)
    if not unique and indexer_sli is not tm.iloc:
        expected = Series([val, val], dtype=object, index=[1, 1])
    tm.assert_series_equal(ser, expected)