import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_multi_assign_broadcasting_rhs(self):
    df = DataFrame({'A': [1, 2, 0, 0, 0], 'B': [0, 0, 0, 10, 11], 'C': [0, 0, 0, 10, 11], 'D': [3, 4, 5, 6, 7]})
    expected = df.copy()
    mask = expected['A'] == 0
    for col in ['A', 'B']:
        expected.loc[mask, col] = df['D']
    df.loc[df['A'] == 0, ['A', 'B']] = df['D'].copy()
    tm.assert_frame_equal(df, expected)