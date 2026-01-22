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
def test_round(self, arr1d):
    dti = self.index_cls(arr1d)
    result = dti.round(freq='2min')
    expected = dti - pd.Timedelta(minutes=1)
    expected = expected._with_freq(None)
    tm.assert_index_equal(result, expected)
    dta = dti._data
    result = dta.round(freq='2min')
    expected = expected._data._with_freq(None)
    tm.assert_datetime_array_equal(result, expected)