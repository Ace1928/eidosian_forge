from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
    rng = timedelta_range('1 days', '10 days')
    expected = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
    rng = tm.box_expected(rng, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    orig_rng = rng
    rng -= two_hours
    tm.assert_equal(rng, expected)
    if box_with_array is not Index:
        tm.assert_equal(orig_rng, expected)