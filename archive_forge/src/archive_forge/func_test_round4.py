import pytest
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_round4(self, tz_naive_fixture):
    index = DatetimeIndex(['2016-10-17 12:00:00.001501031'], dtype='M8[ns]')
    result = index.round('10ns')
    expected = DatetimeIndex(['2016-10-17 12:00:00.001501030'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    ts = '2016-10-17 12:00:00.001501031'
    dti = DatetimeIndex([ts], dtype='M8[ns]')
    with tm.assert_produces_warning(False):
        dti.round('1010ns')