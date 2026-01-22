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
def test_order_of_appearance(self):
    result = pd.unique(Series([2, 1, 3, 3]))
    tm.assert_numpy_array_equal(result, np.array([2, 1, 3], dtype='int64'))
    result = pd.unique(Series([2] + [1] * 5))
    tm.assert_numpy_array_equal(result, np.array([2, 1], dtype='int64'))
    msg = 'unique with argument that is not not a Series, Index,'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.unique(list('aabc'))
    expected = np.array(['a', 'b', 'c'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = pd.unique(Series(Categorical(list('aabc'))))
    expected = Categorical(list('abc'))
    tm.assert_categorical_equal(result, expected)