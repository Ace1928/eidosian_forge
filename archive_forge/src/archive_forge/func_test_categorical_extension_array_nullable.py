from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_extension_array_nullable(self, nulls_fixture):
    arr = pd.arrays.StringArray._from_sequence([nulls_fixture] * 2, dtype=pd.StringDtype())
    result = Categorical(arr)
    assert arr.dtype == result.categories.dtype
    expected = Categorical(Series([pd.NA, pd.NA], dtype=arr.dtype))
    tm.assert_categorical_equal(result, expected)