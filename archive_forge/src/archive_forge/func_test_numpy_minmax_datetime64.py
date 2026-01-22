from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_minmax_datetime64(self):
    dr = date_range(start='2016-01-15', end='2016-01-20')
    assert np.min(dr) == Timestamp('2016-01-15 00:00:00')
    assert np.max(dr) == Timestamp('2016-01-20 00:00:00')
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.min(dr, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.max(dr, out=0)
    assert np.argmin(dr) == 0
    assert np.argmax(dr) == 5
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.argmin(dr, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.argmax(dr, out=0)