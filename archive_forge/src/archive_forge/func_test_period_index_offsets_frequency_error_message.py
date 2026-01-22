import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('freq,freq_depr', [('2M', '2ME'), ('2Q-MAR', '2QE-MAR'), ('2Y-FEB', '2YE-FEB'), ('2M', '2me'), ('2Q-MAR', '2qe-MAR'), ('2Y-FEB', '2yE-feb')])
def test_period_index_offsets_frequency_error_message(self, freq, freq_depr):
    msg = f"for Period, please use '{freq[1:]}' instead of '{freq_depr[1:]}'"
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(['2020-01-01', '2020-01-02'], freq=freq_depr)
    with pytest.raises(ValueError, match=msg):
        period_range(start='2020-01-01', end='2020-01-02', freq=freq_depr)