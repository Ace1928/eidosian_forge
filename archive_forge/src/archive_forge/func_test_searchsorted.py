import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_searchsorted(self, data_for_sorting, as_series):
    data_for_sorting = pd.array([True, False], dtype='boolean')
    b, a = data_for_sorting
    arr = type(data_for_sorting)._from_sequence([a, b])
    if as_series:
        arr = pd.Series(arr)
    assert arr.searchsorted(a) == 0
    assert arr.searchsorted(a, side='right') == 1
    assert arr.searchsorted(b) == 1
    assert arr.searchsorted(b, side='right') == 2
    result = arr.searchsorted(arr.take([0, 1]))
    expected = np.array([0, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    sorter = np.array([1, 0])
    assert data_for_sorting.searchsorted(a, sorter=sorter) == 0