import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('freq', ['M', '2M', '3M'])
def test_pi_cmp_nat_mismatched_freq_raises(self, freq):
    idx1 = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-05'], freq=freq)
    diff = PeriodIndex(['2011-02', '2011-01', '2011-04', 'NaT'], freq='4M')
    msg = f'Invalid comparison between dtype=period\\[{freq}\\] and PeriodArray'
    with pytest.raises(TypeError, match=msg):
        idx1 > diff
    result = idx1 == diff
    expected = np.array([False, False, False, False], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)