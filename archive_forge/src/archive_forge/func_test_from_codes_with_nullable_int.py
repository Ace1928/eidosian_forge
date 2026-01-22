from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_with_nullable_int(self):
    codes = pd.array([0, 1], dtype='Int64')
    categories = ['a', 'b']
    result = Categorical.from_codes(codes, categories=categories)
    expected = Categorical.from_codes(codes.to_numpy(int), categories=categories)
    tm.assert_categorical_equal(result, expected)