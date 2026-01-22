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
def test_astype_assignment_full_replacements(self):
    df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
    df.iloc[:, 0] = df['A'].astype(np.int64)
    expected = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
    df.loc[:, 'A'] = df['A'].astype(np.int64)
    tm.assert_frame_equal(df, expected)