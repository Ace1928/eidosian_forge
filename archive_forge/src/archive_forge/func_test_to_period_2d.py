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
def test_to_period_2d(self, arr1d):
    arr2d = arr1d.reshape(1, -1)
    warn = None if arr1d.tz is None else UserWarning
    with tm.assert_produces_warning(warn):
        result = arr2d.to_period('D')
        expected = arr1d.to_period('D').reshape(1, -1)
    tm.assert_period_array_equal(result, expected)