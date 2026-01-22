import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('na_position, expected', [('last', np.array([2, 0, 1], dtype=np.dtype('intp'))), ('first', np.array([1, 2, 0], dtype=np.dtype('intp')))])
def test_nargsort(self, data_missing_for_sorting, na_position, expected):
    result = nargsort(data_missing_for_sorting, na_position=na_position)
    tm.assert_numpy_array_equal(result, expected)