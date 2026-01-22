import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left_dtype', ['M8[ns]', 'm8[ns]', 'float64', 'int64', 'object'])
@pytest.mark.parametrize('right_dtype', ['M8[ns]', 'm8[ns]', 'float64', 'int64', 'object'])
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    _assert_almost_equal_both(np.array([], dtype=left_dtype), np.array([], dtype=right_dtype), check_dtype=False)