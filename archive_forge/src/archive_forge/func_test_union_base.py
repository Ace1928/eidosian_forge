import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.arm_slow
@pytest.mark.parametrize('klass', [MultiIndex, np.array, Series, list])
def test_union_base(idx, sort, klass):
    first = idx[::-1]
    second = idx[:5]
    if klass is not MultiIndex:
        second = klass(second.values)
    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)