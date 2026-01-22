import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('slice_', [slice(None), slice(0)])
def test_union_sort_other_empty(slice_):
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    other = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    tm.assert_index_equal(other.union(idx), idx)
    tm.assert_index_equal(idx.union(other, sort=False), idx)