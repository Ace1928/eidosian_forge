import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('stamp,expected', [(Timestamp('2018-01-01 23:22:43.325+00:00'), Series(2, name=Timestamp('2018-01-01 23:22:43.325+00:00'))), (Timestamp('2018-01-01 22:33:20.682+01:00'), Series(1, name=Timestamp('2018-01-01 22:33:20.682+01:00')))])
def test_time_zone_aware_index(self, stamp, expected):
    df = DataFrame(data=[1, 2], index=[Timestamp('2018-01-01 21:00:05.001+00:00'), Timestamp('2018-01-01 22:35:10.550+00:00')])
    result = df.asof(stamp)
    tm.assert_series_equal(result, expected)