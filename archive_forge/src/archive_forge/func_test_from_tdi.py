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
def test_from_tdi(self):
    tdi = TimedeltaIndex(['1 Day', '3 Hours'])
    arr = tdi._data
    assert list(arr) == list(tdi)
    tdi2 = pd.Index(arr)
    assert isinstance(tdi2, TimedeltaIndex)
    assert list(tdi2) == list(arr)