import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_union_duplicates(index, request):
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        pytest.skip(f'No duplicates in an empty {type(index).__name__}')
    values = index.unique().values.tolist()
    mi1 = MultiIndex.from_arrays([values, [1] * len(values)])
    mi2 = MultiIndex.from_arrays([[values[0]] + values, [1] * (len(values) + 1)])
    result = mi2.union(mi1)
    expected = mi2.sort_values()
    tm.assert_index_equal(result, expected)
    if is_unsigned_integer_dtype(mi2.levels[0]) and (mi2.get_level_values(0) < 2 ** 63).all():
        expected = expected.set_levels([expected.levels[0].astype(np.int64), expected.levels[1]])
    elif is_float_dtype(mi2.levels[0]):
        expected = expected.set_levels([expected.levels[0].astype(float), expected.levels[1]])
    result = mi1.union(mi2)
    tm.assert_index_equal(result, expected)