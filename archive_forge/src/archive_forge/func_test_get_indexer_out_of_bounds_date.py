from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('target, positions', [([date(9999, 1, 1), Timestamp('2020-01-01')], [-1, 0]), ([Timestamp('2020-01-01'), date(9999, 1, 1)], [0, -1]), ([date(9999, 1, 1), date(9999, 1, 1)], [-1, -1])])
def test_get_indexer_out_of_bounds_date(self, target, positions):
    values = DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02')])
    result = values.get_indexer(target)
    expected = np.array(positions, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)