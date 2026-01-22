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
def test_pivot_table_dropna_categoricals(self, dropna):
    categories = ['a', 'b', 'c', 'd']
    df = DataFrame({'A': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], 'B': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'C': range(9)})
    df['A'] = df['A'].astype(CategoricalDtype(categories, ordered=False))
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.pivot_table(index='B', columns='A', values='C', dropna=dropna)
    expected_columns = Series(['a', 'b', 'c'], name='A')
    expected_columns = expected_columns.astype(CategoricalDtype(categories, ordered=False))
    expected_index = Series([1, 2, 3], name='B')
    expected = DataFrame([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]], index=expected_index, columns=expected_columns)
    if not dropna:
        expected = expected.reindex(columns=Categorical(categories)).astype('float')
    tm.assert_frame_equal(result, expected)