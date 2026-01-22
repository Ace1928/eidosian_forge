import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_count_nonnumeric_types(step):
    cols = ['int', 'float', 'string', 'datetime', 'timedelta', 'periods', 'fl_inf', 'fl_nan', 'str_nan', 'dt_nat', 'periods_nat']
    dt_nat_col = [Timestamp('20170101'), Timestamp('20170203'), Timestamp(None)]
    df = DataFrame({'int': [1, 2, 3], 'float': [4.0, 5.0, 6.0], 'string': list('abc'), 'datetime': date_range('20170101', periods=3), 'timedelta': timedelta_range('1 s', periods=3, freq='s'), 'periods': [Period('2012-01'), Period('2012-02'), Period('2012-03')], 'fl_inf': [1.0, 2.0, np.inf], 'fl_nan': [1.0, 2.0, np.nan], 'str_nan': ['aa', 'bb', np.nan], 'dt_nat': dt_nat_col, 'periods_nat': [Period('2012-01'), Period('2012-02'), Period(None)]}, columns=cols)
    expected = DataFrame({'int': [1.0, 2.0, 2.0], 'float': [1.0, 2.0, 2.0], 'string': [1.0, 2.0, 2.0], 'datetime': [1.0, 2.0, 2.0], 'timedelta': [1.0, 2.0, 2.0], 'periods': [1.0, 2.0, 2.0], 'fl_inf': [1.0, 2.0, 2.0], 'fl_nan': [1.0, 2.0, 1.0], 'str_nan': [1.0, 2.0, 1.0], 'dt_nat': [1.0, 2.0, 1.0], 'periods_nat': [1.0, 2.0, 1.0]}, columns=cols)[::step]
    result = df.rolling(window=2, min_periods=0, step=step).count()
    tm.assert_frame_equal(result, expected)
    result = df.rolling(1, min_periods=0, step=step).count()
    expected = df.notna().astype(float)[::step]
    tm.assert_frame_equal(result, expected)