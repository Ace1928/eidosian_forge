from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('arr,casting_nats', [(TimedeltaIndex(['1 Day', '3 Hours', 'NaT'])._data, (NaT, np.timedelta64('NaT', 'ns'))), (pd.date_range('2000-01-01', periods=3, freq='D')._data, (NaT, np.datetime64('NaT', 'ns'))), (pd.period_range('2000-01-01', periods=3, freq='D')._data, (NaT,))], ids=lambda x: type(x).__name__)
def test_casting_nat_setitem_array(arr, casting_nats):
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]], dtype=arr.dtype)
    for nat in casting_nats:
        arr = arr.copy()
        arr[0] = nat
        tm.assert_equal(arr, expected)