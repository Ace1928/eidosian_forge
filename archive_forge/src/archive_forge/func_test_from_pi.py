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
def test_from_pi(self, arr1d):
    pi = self.index_cls(arr1d)
    arr = arr1d
    assert list(arr) == list(pi)
    pi2 = pd.Index(arr)
    assert isinstance(pi2, PeriodIndex)
    assert list(pi2) == list(arr)