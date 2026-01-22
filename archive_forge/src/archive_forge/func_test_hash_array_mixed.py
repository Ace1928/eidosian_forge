import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('dtype', ['U', object])
def test_hash_array_mixed(dtype):
    result1 = hash_array(np.array(['3', '4', 'All']))
    result2 = hash_array(np.array([3, 4, 'All'], dtype=dtype))
    tm.assert_numpy_array_equal(result1, result2)