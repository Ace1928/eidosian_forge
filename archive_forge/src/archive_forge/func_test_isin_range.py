import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('base', [RangeIndex(0, 2), Index([0, 1])])
def test_isin_range(self, base):
    values = RangeIndex(0, 1)
    result = base.isin(values)
    expected = np.array([True, False])
    tm.assert_numpy_array_equal(result, expected)