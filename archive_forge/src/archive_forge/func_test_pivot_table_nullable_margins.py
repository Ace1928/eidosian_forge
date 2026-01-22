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
def test_pivot_table_nullable_margins(self):
    df = DataFrame({'a': 'A', 'b': [1, 2], 'sales': Series([10, 11], dtype='Int64')})
    result = df.pivot_table(index='b', columns='a', margins=True, aggfunc='sum')
    expected = DataFrame([[10, 10], [11, 11], [21, 21]], index=Index([1, 2, 'All'], name='b'), columns=MultiIndex.from_tuples([('sales', 'A'), ('sales', 'All')], names=[None, 'a']), dtype='Int64')
    tm.assert_frame_equal(result, expected)