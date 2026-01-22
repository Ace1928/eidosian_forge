from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_to_period(self, datetime_index, freqstr):
    dti = datetime_index
    arr = dti._data
    freqstr = freq_to_period_freqstr(1, freqstr)
    expected = dti.to_period(freq=freqstr)
    result = arr.to_period(freq=freqstr)
    assert isinstance(result, PeriodArray)
    tm.assert_equal(result, expected._data)