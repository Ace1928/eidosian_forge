from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_maybe_cast_slice_bounds_empty(self):
    empty_idx = date_range(freq='1h', periods=0, end='2015')
    right = empty_idx._maybe_cast_slice_bound('2015-01-02', 'right')
    exp = Timestamp('2015-01-02 23:59:59.999999999')
    assert right == exp
    left = empty_idx._maybe_cast_slice_bound('2015-01-02', 'left')
    exp = Timestamp('2015-01-02 00:00:00')
    assert left == exp