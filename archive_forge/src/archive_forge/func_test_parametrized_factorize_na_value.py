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
@pytest.mark.parametrize('data, na_value', [(np.array([0, 1, 0, 2], dtype='u8'), 0), (np.array([1, 0, 1, 2], dtype='u8'), 1), (np.array([-2 ** 63, 1, -2 ** 63, 0], dtype='i8'), -2 ** 63), (np.array([1, -2 ** 63, 1, 0], dtype='i8'), 1), (np.array(['a', '', 'a', 'b'], dtype=object), 'a'), (np.array([(), ('a', 1), (), ('a', 2)], dtype=object), ()), (np.array([('a', 1), (), ('a', 1), ('a', 2)], dtype=object), ('a', 1))])
def test_parametrized_factorize_na_value(self, data, na_value):
    codes, uniques = algos.factorize_array(data, na_value=na_value)
    expected_uniques = data[[1, 3]]
    expected_codes = np.array([-1, 0, -1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_numpy_array_equal(uniques, expected_uniques)