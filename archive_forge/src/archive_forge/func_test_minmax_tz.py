from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_minmax_tz(self, tz_naive_fixture):
    tz = tz_naive_fixture
    idx1 = DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz=tz)
    assert idx1.is_monotonic_increasing
    idx2 = DatetimeIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], tz=tz)
    assert not idx2.is_monotonic_increasing
    for idx in [idx1, idx2]:
        assert idx.min() == Timestamp('2011-01-01', tz=tz)
        assert idx.max() == Timestamp('2011-01-03', tz=tz)
        assert idx.argmin() == 0
        assert idx.argmax() == 2