from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('klass', [lambda x: np.array(x, dtype=object), list])
def test_construction_with_null(self, klass, nulls_fixture):
    values = klass(['a', nulls_fixture, 'b'])
    result = Categorical(values)
    dtype = CategoricalDtype(['a', 'b'])
    codes = [0, -1, 1]
    expected = Categorical.from_codes(codes=codes, dtype=dtype)
    tm.assert_categorical_equal(result, expected)