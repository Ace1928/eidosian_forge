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
def test_margins_dtype(self, data):
    df = data.copy()
    df[['D', 'E', 'F']] = np.arange(len(df) * 3).reshape(len(df), 3).astype('i8')
    mi_val = list(product(['bar', 'foo'], ['one', 'two'])) + [('All', '')]
    mi = MultiIndex.from_tuples(mi_val, names=('A', 'B'))
    expected = DataFrame({'dull': [12, 21, 3, 9, 45], 'shiny': [33, 0, 36, 51, 120]}, index=mi).rename_axis('C', axis=1)
    expected['All'] = expected['dull'] + expected['shiny']
    result = df.pivot_table(values='D', index=['A', 'B'], columns='C', margins=True, aggfunc='sum', fill_value=0)
    tm.assert_frame_equal(expected, result)