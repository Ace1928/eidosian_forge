from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_from_categorical_with_dtype(self):
    dtype = CategoricalDtype(['a', 'b', 'c'], ordered=True)
    values = Categorical(['a', 'b', 'd'])
    result = Categorical(values, dtype=dtype)
    expected = Categorical(['a', 'b', 'd'], categories=['a', 'b', 'c'], ordered=True)
    tm.assert_categorical_equal(result, expected)