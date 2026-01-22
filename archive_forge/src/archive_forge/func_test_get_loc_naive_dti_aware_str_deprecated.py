import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_naive_dti_aware_str_deprecated(self):
    ts = Timestamp('20130101')._value
    dti = pd.DatetimeIndex([ts + 50 + i for i in range(100)])
    ser = Series(range(100), index=dti)
    key = '2013-01-01 00:00:00.000000050+0000'
    msg = re.escape(repr(key))
    with pytest.raises(KeyError, match=msg):
        ser[key]
    with pytest.raises(KeyError, match=msg):
        dti.get_loc(key)