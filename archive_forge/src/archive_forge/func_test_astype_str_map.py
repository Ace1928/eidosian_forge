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
@pytest.mark.parametrize('dtype', [str, np.str_])
@pytest.mark.parametrize('series', [Series([string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)]), Series([string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0])])
def test_astype_str_map(self, dtype, series, using_infer_string):
    result = series.astype(dtype)
    expected = series.map(str)
    if using_infer_string:
        expected = expected.astype(object)
    tm.assert_series_equal(result, expected)