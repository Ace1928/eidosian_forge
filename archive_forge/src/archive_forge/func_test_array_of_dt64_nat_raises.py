from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_array_of_dt64_nat_raises(self):
    nat = np.datetime64('NaT', 'ns')
    arr = np.array([nat], dtype=object)
    msg = 'Invalid type for timedelta scalar'
    with pytest.raises(TypeError, match=msg):
        TimedeltaIndex(arr)
    with pytest.raises(TypeError, match=msg):
        TimedeltaArray._from_sequence(arr, dtype='m8[ns]')
    with pytest.raises(TypeError, match=msg):
        to_timedelta(arr)