import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('mult', [1, 2, 3, 4, 5])
def test_constructor_freq_mult_dti_compat_month(self, mult):
    pidx = period_range(start='2014-04-01', freq=f'{mult}M', periods=10)
    expected = date_range(start='2014-04-01', freq=f'{mult}ME', periods=10).to_period(f'{mult}M')
    tm.assert_index_equal(pidx, expected)