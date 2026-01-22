from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('dtype', (np.float64, object, 'M8[ns]'))
def test_value_counts_normalized(self, dtype):
    s = Series([1] * 2 + [2] * 3 + [np.nan] * 5)
    s_typed = s.astype(dtype)
    result = s_typed.value_counts(normalize=True, dropna=False)
    expected = Series([0.5, 0.3, 0.2], index=Series([np.nan, 2.0, 1.0], dtype=dtype), name='proportion')
    tm.assert_series_equal(result, expected)
    result = s_typed.value_counts(normalize=True, dropna=True)
    expected = Series([0.6, 0.4], index=Series([2.0, 1.0], dtype=dtype), name='proportion')
    tm.assert_series_equal(result, expected)