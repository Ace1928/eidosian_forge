import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_value_counts_period(self):
    values = [pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M'), pd.Period('2011-03', freq='M'), pd.Period('2011-01', freq='M'), pd.Period('2011-01', freq='M'), pd.Period('2011-03', freq='M')]
    exp_idx = pd.PeriodIndex(['2011-01', '2011-03', '2011-02'], freq='M', name='xxx')
    exp = Series([3, 2, 1], index=exp_idx, name='count')
    ser = Series(values, name='xxx')
    tm.assert_series_equal(ser.value_counts(), exp)
    idx = pd.PeriodIndex(values, name='xxx')
    tm.assert_series_equal(idx.value_counts(), exp)
    exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name='proportion')
    tm.assert_series_equal(ser.value_counts(normalize=True), exp)
    tm.assert_series_equal(idx.value_counts(normalize=True), exp)