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
@pytest.mark.parametrize('columns, aggfunc, values, expected_columns', [('A', 'mean', [[5.5, 5.5, 2.2, 2.2], [8.0, 8.0, 4.4, 4.4]], Index(['bar', 'All', 'foo', 'All'], name='A')), (['A', 'B'], 'sum', [[9, 13, 22, 5, 6, 11], [14, 18, 32, 11, 11, 22]], MultiIndex.from_tuples([('bar', 'one'), ('bar', 'two'), ('bar', 'All'), ('foo', 'one'), ('foo', 'two'), ('foo', 'All')], names=['A', 'B']))])
def test_margin_with_only_columns_defined(self, columns, aggfunc, values, expected_columns):
    df = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar'], 'B': ['one', 'one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'], 'C': ['small', 'large', 'large', 'small', 'small', 'large', 'small', 'small', 'large'], 'D': [1, 2, 2, 3, 3, 4, 5, 6, 7], 'E': [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    if aggfunc != 'sum':
        msg = re.escape('agg function failed [how->mean,dtype->')
        with pytest.raises(TypeError, match=msg):
            df.pivot_table(columns=columns, margins=True, aggfunc=aggfunc)
    if 'B' not in columns:
        df = df.drop(columns='B')
    result = df.drop(columns='C').pivot_table(columns=columns, margins=True, aggfunc=aggfunc)
    expected = DataFrame(values, index=Index(['D', 'E']), columns=expected_columns)
    tm.assert_frame_equal(result, expected)