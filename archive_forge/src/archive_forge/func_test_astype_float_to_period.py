from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_float_to_period(self):
    result = Series([np.nan]).astype('period[D]')
    expected = Series([NaT], dtype='period[D]')
    tm.assert_series_equal(result, expected)