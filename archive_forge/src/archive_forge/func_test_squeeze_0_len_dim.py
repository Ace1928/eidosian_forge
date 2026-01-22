from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_squeeze_0_len_dim(self):
    empty_series = Series([], name='five', dtype=np.float64)
    empty_frame = DataFrame([empty_series])
    tm.assert_series_equal(empty_series, empty_series.squeeze())
    tm.assert_series_equal(empty_series, empty_frame.squeeze())