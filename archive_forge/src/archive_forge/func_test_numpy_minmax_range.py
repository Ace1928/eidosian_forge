from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_minmax_range(self):
    idx = RangeIndex(0, 10, 3)
    result = np.max(idx)
    assert result == 9
    result = np.min(idx)
    assert result == 0
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        np.min(idx, out=0)
    with pytest.raises(ValueError, match=errmsg):
        np.max(idx, out=0)