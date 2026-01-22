from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_div_td_scalar(self, tda):
    other = timedelta(seconds=1)
    result = tda / other
    expected = tda._ndarray / np.timedelta64(1, 's')
    tm.assert_numpy_array_equal(result, expected)