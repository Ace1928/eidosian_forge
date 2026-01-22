import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
@pytest.mark.parametrize('method', ['backfill', 'pad'])
def test_fillna_2d_method(self, data_missing, method):
    arr = data_missing.repeat(2).reshape(2, 2)
    assert arr[0].isna().all()
    assert not arr[1].isna().any()
    result = arr._pad_or_backfill(method=method, limit=None)
    expected = data_missing._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
    tm.assert_extension_array_equal(result, expected)
    arr2 = arr[::-1]
    assert not arr2[0].isna().any()
    assert arr2[1].isna().all()
    result2 = arr2._pad_or_backfill(method=method, limit=None)
    expected2 = data_missing[::-1]._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
    tm.assert_extension_array_equal(result2, expected2)