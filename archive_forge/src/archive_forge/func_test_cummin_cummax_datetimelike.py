import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ts', [pd.Timedelta(0), pd.Timestamp('1999-12-31'), pd.Timestamp('1999-12-31').tz_localize('US/Pacific')])
@pytest.mark.parametrize('method, skipna, exp_tdi', [['cummax', True, ['NaT', '2 days', 'NaT', '2 days', 'NaT', '3 days']], ['cummin', True, ['NaT', '2 days', 'NaT', '1 days', 'NaT', '1 days']], ['cummax', False, ['NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT']], ['cummin', False, ['NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT']]])
def test_cummin_cummax_datetimelike(self, ts, method, skipna, exp_tdi):
    tdi = pd.to_timedelta(['NaT', '2 days', 'NaT', '1 days', 'NaT', '3 days'])
    ser = pd.Series(tdi + ts)
    exp_tdi = pd.to_timedelta(exp_tdi)
    expected = pd.Series(exp_tdi + ts)
    result = getattr(ser, method)(skipna=skipna)
    tm.assert_series_equal(expected, result)