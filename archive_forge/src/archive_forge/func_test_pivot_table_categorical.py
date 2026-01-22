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
def test_pivot_table_categorical(self):
    cat1 = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'], ordered=True)
    cat2 = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'], ordered=True)
    df = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pivot_table(df, values='values', index=['A', 'B'], dropna=True)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=['A', 'B'])
    expected = DataFrame({'values': [1.0, 2.0, 3.0, 4.0]}, index=exp_index)
    tm.assert_frame_equal(result, expected)