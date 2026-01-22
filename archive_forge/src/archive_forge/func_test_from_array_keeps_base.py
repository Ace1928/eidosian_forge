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
def test_from_array_keeps_base(self):
    arr = np.array(['2000-01-01', '2000-01-02'], dtype='M8[ns]')
    dta = DatetimeArray._from_sequence(arr)
    assert dta._ndarray is arr
    dta = DatetimeArray._from_sequence(arr[:0])
    assert dta._ndarray.base is arr