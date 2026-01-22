from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_range_overflows(self):
    rng = range(2 ** 63, 2 ** 63 + 4)
    ser = Series(rng)
    expected = Series(list(rng))
    tm.assert_series_equal(ser, expected)
    assert list(ser) == list(rng)
    assert ser.dtype == np.uint64
    rng2 = range(2 ** 63 + 4, 2 ** 63, -1)
    ser2 = Series(rng2)
    expected2 = Series(list(rng2))
    tm.assert_series_equal(ser2, expected2)
    assert list(ser2) == list(rng2)
    assert ser2.dtype == np.uint64
    rng3 = range(-2 ** 63, -2 ** 63 - 4, -1)
    ser3 = Series(rng3)
    expected3 = Series(list(rng3))
    tm.assert_series_equal(ser3, expected3)
    assert list(ser3) == list(rng3)
    assert ser3.dtype == object
    rng4 = range(2 ** 73, 2 ** 73 + 4)
    ser4 = Series(rng4)
    expected4 = Series(list(rng4))
    tm.assert_series_equal(ser4, expected4)
    assert list(ser4) == list(rng4)
    assert ser4.dtype == object