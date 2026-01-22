import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('other', [Index(['three', 'one', 'two']), Index(['one']), Index(['one', 'three'])])
def test_join_level(idx, other, join_type):
    join_index, lidx, ridx = other.join(idx, how=join_type, level='second', return_indexers=True)
    exp_level = other.join(idx.levels[1], how=join_type)
    assert join_index.levels[0].equals(idx.levels[0])
    assert join_index.levels[1].equals(exp_level)
    mask = np.array([x[1] in exp_level for x in idx], dtype=bool)
    exp_values = idx.values[mask]
    tm.assert_numpy_array_equal(join_index.values, exp_values)
    if join_type in ('outer', 'inner'):
        join_index2, ridx2, lidx2 = idx.join(other, how=join_type, level='second', return_indexers=True)
        assert join_index.equals(join_index2)
        tm.assert_numpy_array_equal(lidx, lidx2)
        tm.assert_numpy_array_equal(ridx, ridx2)
        tm.assert_numpy_array_equal(join_index2.values, exp_values)