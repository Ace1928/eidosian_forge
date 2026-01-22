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
@pytest.mark.parametrize('klass', [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass):
    strings = ['2020Q1', '2020Q2'] * 2
    data = klass(strings)
    result = PeriodIndex(data, freq='Q')
    expected = PeriodIndex([Period(s) for s in strings])
    tm.assert_index_equal(result, expected)