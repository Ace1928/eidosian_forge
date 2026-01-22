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
def test_int64_overflow_sort_false_order(self, left_right):
    left, right = left_right
    out = merge(left, right, how='left', sort=False)
    tm.assert_frame_equal(left, out[left.columns.tolist()])
    out = merge(right, left, how='left', sort=False)
    tm.assert_frame_equal(right, out[right.columns.tolist()])