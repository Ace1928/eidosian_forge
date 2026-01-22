from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_minmax_timedelta64(self):
    idx1 = TimedeltaIndex(['1 days', '2 days', '3 days'])
    assert idx1.is_monotonic_increasing
    idx2 = TimedeltaIndex(['1 days', np.nan, '3 days', 'NaT'])
    assert not idx2.is_monotonic_increasing
    for idx in [idx1, idx2]:
        assert idx.min() == Timedelta('1 days')
        assert idx.max() == Timedelta('3 days')
        assert idx.argmin() == 0
        assert idx.argmax() == 2