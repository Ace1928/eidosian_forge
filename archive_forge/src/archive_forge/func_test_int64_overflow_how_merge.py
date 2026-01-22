from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.slow
@pytest.mark.parametrize('how', ['left', 'right', 'outer', 'inner'])
def test_int64_overflow_how_merge(self, left_right, how):
    left, right = left_right
    out = merge(left, right, how='outer')
    out.sort_values(out.columns.tolist(), inplace=True)
    out.index = np.arange(len(out))
    tm.assert_frame_equal(out, merge(left, right, how=how, sort=True))