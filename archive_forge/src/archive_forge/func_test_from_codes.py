from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes(self):
    dtype = CategoricalDtype(categories=['a', 'b', 'c'])
    exp = Categorical(['a', 'b', 'c'], ordered=False)
    res = Categorical.from_codes([0, 1, 2], categories=dtype.categories)
    tm.assert_categorical_equal(exp, res)
    res = Categorical.from_codes([0, 1, 2], dtype=dtype)
    tm.assert_categorical_equal(exp, res)