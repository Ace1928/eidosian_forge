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
def test_series_factorize_use_na_sentinel_false(self):
    values = np.array([1, 2, 1, np.nan])
    ser = Series(values)
    codes, uniques = ser.factorize(use_na_sentinel=False)
    expected_codes = np.array([0, 1, 0, 2], dtype=np.intp)
    expected_uniques = Index([1.0, 2.0, np.nan])
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_index_equal(uniques, expected_uniques)