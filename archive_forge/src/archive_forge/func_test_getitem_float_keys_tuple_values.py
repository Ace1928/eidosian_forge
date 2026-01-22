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
def test_getitem_float_keys_tuple_values(self):
    ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.1, 0.2], name='foo')
    result = ser[0.0]
    assert result == (1, 1)
    expected = Series([(1, 1), (2, 2)], index=[0.0, 0.0], name='foo')
    ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.0, 0.2], name='foo')
    result = ser[0.0]
    tm.assert_series_equal(result, expected)