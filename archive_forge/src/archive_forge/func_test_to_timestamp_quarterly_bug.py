from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_quarterly_bug(self):
    years = np.arange(1960, 2000).repeat(4)
    quarters = np.tile(list(range(1, 5)), 40)
    pindex = PeriodIndex.from_fields(year=years, quarter=quarters)
    stamps = pindex.to_timestamp('D', 'end')
    expected = DatetimeIndex([x.to_timestamp('D', 'end') for x in pindex])
    tm.assert_index_equal(stamps, expected)
    assert stamps.freq == expected.freq