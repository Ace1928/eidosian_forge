import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_period_array_freq_mismatch():
    arr = period_array(['2000', '2001'], freq='D')
    with pytest.raises(IncompatibleFrequency, match='freq'):
        PeriodArray(arr, dtype='period[M]')
    dtype = pd.PeriodDtype(pd.tseries.offsets.MonthEnd())
    with pytest.raises(IncompatibleFrequency, match='freq'):
        PeriodArray(arr, dtype=dtype)