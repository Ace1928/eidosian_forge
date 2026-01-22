import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
import pandas._testing as tm
def test_get_resolution_non_nano_data():
    arr = np.array([1], dtype=np.int64)
    res = get_resolution(arr, None, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US
    res = get_resolution(arr, pytz.UTC, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US