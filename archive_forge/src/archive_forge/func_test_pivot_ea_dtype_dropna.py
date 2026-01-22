from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('dropna', [True, False])
def test_pivot_ea_dtype_dropna(self, dropna):
    df = DataFrame({'x': 'a', 'y': 'b', 'age': Series([20, 40], dtype='Int64')})
    result = df.pivot_table(index='x', columns='y', values='age', aggfunc='mean', dropna=dropna)
    expected = DataFrame([[30]], index=Index(['a'], name='x'), columns=Index(['b'], name='y'), dtype='Float64')
    tm.assert_frame_equal(result, expected)