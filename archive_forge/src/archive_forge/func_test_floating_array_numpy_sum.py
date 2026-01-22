import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('values, expected', [([1, 2, 3], 6.0), ([1, 2, 3, None], 6.0), ([None], 0.0)])
def test_floating_array_numpy_sum(values, expected):
    arr = pd.array(values, dtype='Float64')
    result = np.sum(arr)
    assert result == expected