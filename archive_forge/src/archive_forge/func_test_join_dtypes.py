import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [0, 5])
def test_join_dtypes(any_numeric_ea_dtype, val):
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [3, 4]])
    midx2 = MultiIndex.from_arrays([Series([1, val, val], dtype=any_numeric_ea_dtype), [3, 4, 4]])
    result = midx.join(midx2, how='outer')
    expected = MultiIndex.from_arrays([Series([val, val, 1, 2], dtype=any_numeric_ea_dtype), [4, 4, 3, 4]]).sort_values()
    tm.assert_index_equal(result, expected)