import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('case', [[4, 2, 0, -2], [2, 2, 1, 0], [0, 1, 2, 1]])
def test_maybe_indices_to_slice_both_edges_not_slice(self, case):
    target = np.arange(10)
    indices = np.array(case, dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert not isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(maybe_slice, indices)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])