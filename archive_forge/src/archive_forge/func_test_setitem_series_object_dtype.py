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
@pytest.mark.parametrize('indexer', [tm.loc, tm.at])
@pytest.mark.parametrize('ser_index', [0, 1])
def test_setitem_series_object_dtype(self, indexer, ser_index):
    ser = Series([0, 0], dtype='object')
    idxr = indexer(ser)
    idxr[0] = Series([42], index=[ser_index])
    expected = Series([Series([42], index=[ser_index]), 0], dtype='object')
    tm.assert_series_equal(ser, expected)