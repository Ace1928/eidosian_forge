from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_assignment_series_alignment():
    ser = Series(range(10))
    idx = np.array([2, 4, 9])
    ser[idx] = Series([10, 11, 12])
    expected = Series([0, 1, 10, 3, 11, 5, 6, 7, 8, 12])
    tm.assert_series_equal(ser, expected)