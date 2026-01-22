import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_use_start_freq(self):
    msg1 = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg1):
        p = Period('4/2/2012', freq='B')
    msg2 = 'PeriodDtype\\[B\\] is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        expected = period_range(start='4/2/2012', periods=10, freq='B')
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        index = period_range(start=p, periods=10)
    tm.assert_index_equal(index, expected)