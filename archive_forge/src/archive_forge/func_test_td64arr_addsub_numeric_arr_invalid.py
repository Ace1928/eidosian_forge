from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('vec', [np.array([1, 2, 3]), Index([1, 2, 3]), Series([1, 2, 3]), DataFrame([[1, 2, 3]])], ids=lambda x: type(x).__name__)
def test_td64arr_addsub_numeric_arr_invalid(self, box_with_array, vec, any_real_numpy_dtype):
    tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
    tdarr = tm.box_expected(tdser, box_with_array)
    vector = vec.astype(any_real_numpy_dtype)
    assert_invalid_addsub_type(tdarr, vector)