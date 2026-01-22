from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_readonly():
    a = np.array(range(3))
    b = ['a', 'b']
    expected = MultiIndex.from_product([a, b])
    a.setflags(write=False)
    result = MultiIndex.from_product([a, b])
    tm.assert_index_equal(result, expected)