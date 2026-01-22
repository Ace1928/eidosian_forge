import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('val', [4, 1])
def test_union_keep_ea_dtype(any_numeric_ea_dtype, val):
    arr1 = Series([val, 2], dtype=any_numeric_ea_dtype)
    arr2 = Series([2, 1], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [1, 2]], names=['a', None])
    midx2 = MultiIndex.from_arrays([arr2, [2, 1]])
    result = midx.union(midx2)
    if val == 4:
        expected = MultiIndex.from_arrays([Series([1, 2, 4], dtype=any_numeric_ea_dtype), [1, 2, 1]])
    else:
        expected = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [1, 2]])
    tm.assert_index_equal(result, expected)