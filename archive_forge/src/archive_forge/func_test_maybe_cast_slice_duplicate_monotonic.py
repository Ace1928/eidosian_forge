from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_maybe_cast_slice_duplicate_monotonic(self):
    idx = DatetimeIndex(['2017', '2017'])
    result = idx._maybe_cast_slice_bound('2017-01-01', 'left')
    expected = Timestamp('2017-01-01')
    assert result == expected