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
def test_getitem_slice_integers(self):
    ser = Series(np.random.default_rng(2).standard_normal(8), index=[2, 4, 6, 8, 10, 12, 14, 16])
    result = ser[:4]
    expected = Series(ser.values[:4], index=[2, 4, 6, 8])
    tm.assert_series_equal(result, expected)