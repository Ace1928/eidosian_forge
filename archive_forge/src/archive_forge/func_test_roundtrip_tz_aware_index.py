from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_roundtrip_tz_aware_index(setup_path, unit):
    ts = Timestamp('2000-01-01 01:00:00', tz='US/Eastern')
    dti = DatetimeIndex([ts]).as_unit(unit)
    df = DataFrame(data=[0], index=dti)
    with ensure_clean_store(setup_path) as store:
        store.put('frame', df, format='fixed')
        recons = store['frame']
        tm.assert_frame_equal(recons, df)
    value = recons.index[0]._value
    denom = {'ns': 1, 'us': 1000, 'ms': 10 ** 6, 's': 10 ** 9}[unit]
    assert value == 946706400000000000 // denom