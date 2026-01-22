import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['UTC', 'US/Eastern', 'Asia/Tokyo', 'EST5EDT'])
def test_concatlike_datetimetz_short(self, tz):
    ix1 = pd.date_range(start='2014-07-15', end='2014-07-17', freq='D', tz=tz)
    ix2 = pd.DatetimeIndex(['2014-07-11', '2014-07-21'], tz=tz)
    df1 = DataFrame(0, index=ix1, columns=['A', 'B'])
    df2 = DataFrame(0, index=ix2, columns=['A', 'B'])
    exp_idx = pd.DatetimeIndex(['2014-07-15', '2014-07-16', '2014-07-17', '2014-07-11', '2014-07-21'], tz=tz).as_unit('ns')
    exp = DataFrame(0, index=exp_idx, columns=['A', 'B'])
    tm.assert_frame_equal(df1._append(df2), exp)
    tm.assert_frame_equal(pd.concat([df1, df2]), exp)