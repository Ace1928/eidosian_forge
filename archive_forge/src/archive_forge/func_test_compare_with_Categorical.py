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
@pytest.mark.parametrize('result', [pd.date_range('2020', periods=3), pd.date_range('2020', periods=3, tz='UTC'), pd.timedelta_range('0 days', periods=3), pd.period_range('2020Q1', periods=3, freq='Q')])
def test_compare_with_Categorical(self, result):
    expected = pd.Categorical(result)
    assert all(result == expected)
    assert not any(result != expected)