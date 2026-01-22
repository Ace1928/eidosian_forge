import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_repeat_range2(self, tz_naive_fixture, unit):
    tz = tz_naive_fixture
    index = date_range('2001-01-01', periods=2, freq='D', tz=tz, unit=unit)
    exp = DatetimeIndex(['2001-01-01', '2001-01-01', '2001-01-02', '2001-01-02'], tz=tz).as_unit(unit)
    for res in [index.repeat(2), np.repeat(index, 2)]:
        tm.assert_index_equal(res, exp)
        assert res.freq is None