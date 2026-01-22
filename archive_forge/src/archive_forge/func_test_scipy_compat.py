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
@pytest.mark.parametrize('arr', [[np.nan, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 3, np.nan], [4.0, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 4.0, np.nan]])
def test_scipy_compat(self, arr):
    sp_stats = pytest.importorskip('scipy.stats')
    arr = np.array(arr)
    mask = ~np.isfinite(arr)
    arr = arr.copy()
    result = libalgos.rank_1d(arr)
    arr[mask] = np.inf
    exp = sp_stats.rankdata(arr)
    exp[mask] = np.nan
    tm.assert_almost_equal(result, exp)