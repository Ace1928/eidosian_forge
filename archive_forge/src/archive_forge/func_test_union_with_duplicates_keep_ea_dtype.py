import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('dupe_val', [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val, any_numeric_ea_dtype):
    mi1 = MultiIndex.from_arrays([Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype), Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype)])
    mi2 = MultiIndex.from_arrays([Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype), Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype)])
    result = mi1.union(mi2)
    expected = MultiIndex.from_arrays([Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype), Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype)])
    tm.assert_index_equal(result, expected)