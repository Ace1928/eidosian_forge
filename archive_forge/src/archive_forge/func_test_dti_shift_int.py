from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_shift_int(self, unit):
    rng = date_range('1/1/2000', periods=20, unit=unit)
    result = rng + 5 * rng.freq
    expected = rng.shift(5)
    tm.assert_index_equal(result, expected)
    result = rng - 5 * rng.freq
    expected = rng.shift(-5)
    tm.assert_index_equal(result, expected)