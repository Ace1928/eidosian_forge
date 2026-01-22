import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_mixed_freq_raises(self):
    msg = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        end_intv = Period('2005-05-01', 'B')
    msg = "'w' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        vals = [end_intv, Period('2006-12-31', 'w')]
    msg = 'Input has different freq=W-SUN from PeriodIndex\\(freq=B\\)'
    depr_msg = 'PeriodDtype\\[B\\] is deprecated'
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            PeriodIndex(vals)
    vals = np.array(vals)
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            PeriodIndex(vals)