import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('start, end, freq, expected_periods', [('1D', '10D', '2D', (10 - 1) // 2 + 1), ('2D', '30D', '3D', (30 - 2) // 3 + 1), ('2s', '50s', '5s', (50 - 2) // 5 + 1), ('4D', '16D', '3D', (16 - 4) // 3 + 1), ('8D', '16D', '40s', (16 * 3600 * 24 - 8 * 3600 * 24) // 40 + 1)])
def test_timedelta_range_freq_divide_end(self, start, end, freq, expected_periods):
    res = timedelta_range(start=start, end=end, freq=freq)
    assert Timedelta(start) == res[0]
    assert Timedelta(end) >= res[-1]
    assert len(res) == expected_periods