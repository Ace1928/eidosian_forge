from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_sub_nat_retain_unit(self):
    ser = pd.to_timedelta(Series(['00:00:01'])).astype('m8[s]')
    result = ser - NaT
    expected = Series([NaT], dtype='m8[s]')
    tm.assert_series_equal(result, expected)