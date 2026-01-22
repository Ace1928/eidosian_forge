import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('data, freq, expected', [([pd.Period('2017', 'D')], None, [17167]), ([pd.Period('2017', 'D')], 'D', [17167]), ([2017], 'D', [17167]), (['2017'], 'D', [17167]), ([pd.Period('2017', 'D')], pd.tseries.offsets.Day(), [17167]), ([pd.Period('2017', 'D'), None], None, [17167, iNaT]), (pd.Series(pd.date_range('2017', periods=3)), None, [17167, 17168, 17169]), (pd.date_range('2017', periods=3), None, [17167, 17168, 17169]), (pd.period_range('2017', periods=4, freq='Q'), None, [188, 189, 190, 191])])
def test_period_array_ok(data, freq, expected):
    result = period_array(data, freq=freq).asi8
    expected = np.asarray(expected, dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)