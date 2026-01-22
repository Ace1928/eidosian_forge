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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't multiply arrow strings")
def test_multi_assign(self):
    df = DataFrame({'FC': ['a', 'b', 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': list(range(6)), 'col2': list(range(6, 12))}).astype({'col2': 'float64'})
    df.iloc[1, 0] = np.nan
    df2 = df.copy()
    mask = ~df2.FC.isna()
    cols = ['col1', 'col2']
    dft = df2 * 2
    dft.iloc[3, 3] = np.nan
    expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': Series([0, 1, 4, 6, 8, 10]), 'col2': [12, 7, 16, np.nan, 20, 22]})
    df2.loc[mask, cols] = dft.loc[mask, cols]
    tm.assert_frame_equal(df2, expected)
    expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': [0, 1, 4, 6, 8, 10], 'col2': [12, 7, 16, np.nan, 20, 22]})
    df2 = df.copy()
    df2.loc[mask, cols] = dft.loc[mask, cols].values
    tm.assert_frame_equal(df2, expected)