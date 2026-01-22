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
def test_factorize_nan(self):
    key = np.array([1, 2, 1, np.nan], dtype='O')
    rizer = ht.ObjectFactorizer(len(key))
    for na_sentinel in (-1, 20):
        ids = rizer.factorize(key, na_sentinel=na_sentinel)
        expected = np.array([0, 1, 0, na_sentinel], dtype=np.intp)
        assert len(set(key)) == len(set(expected))
        tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)
        tm.assert_numpy_array_equal(ids, expected)