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
def test_categorical_isin(self):
    vals = np.array([0, 1, 2, 0])
    cats = ['a', 'b', 'c']
    cat = Categorical([1]).from_codes(vals, cats)
    other = Categorical([1]).from_codes(np.array([0, 1]), cats)
    expected = np.array([True, True, False, True])
    result = algos.isin(cat, other)
    tm.assert_numpy_array_equal(expected, result)