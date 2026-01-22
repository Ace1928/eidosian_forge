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
def test_same_nan_is_in_large(self):
    s = np.tile(1.0, 1000001)
    s[0] = np.nan
    result = algos.isin(s, np.array([np.nan, 1]))
    expected = np.ones(len(s), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)