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
def test_factorizer_object_with_nan(self):
    data = np.array([1, 2, 3, 1, np.nan])
    rizer = ht.ObjectFactorizer(len(data))
    result = rizer.factorize(data.astype(object))
    expected = np.array([0, 1, 2, 0, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    expected_uniques = np.array([1, 2, 3], dtype=object)
    tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)