from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_astype_freq_conversion(self):
    tdi = timedelta_range('1 Day', periods=30)
    res = tdi.astype('m8[s]')
    exp_values = np.asarray(tdi).astype('m8[s]')
    exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype, freq=tdi.freq)
    expected = Index(exp_tda)
    assert expected.dtype == 'm8[s]'
    tm.assert_index_equal(res, expected)
    res = tdi._data.astype('m8[s]')
    tm.assert_equal(res, expected._values)
    res = tdi.to_series().astype('m8[s]')
    tm.assert_equal(res._values, expected._values._with_freq(None))