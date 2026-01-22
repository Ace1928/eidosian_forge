from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ind1', [[True], Index([True])])
@pytest.mark.parametrize('ind2', [[False], Index([False])])
def test_getitem_bool_index_single(ind1, ind2):
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)
    expected = MultiIndex(levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)], codes=[[], []])
    tm.assert_index_equal(idx[ind2], expected)