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
def test_loc_setitem_indexing_assignment_dict_already_exists(self):
    index = Index([-5, 0, 5], name='z')
    df = DataFrame({'x': [1, 2, 6], 'y': [2, 2, 8]}, index=index)
    expected = df.copy()
    rhs = {'x': 9, 'y': 99}
    df.loc[5] = rhs
    expected.loc[5] = [9, 99]
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'x': [1, 2, 6], 'y': [2.0, 2.0, 8.0]}, index=index)
    df.loc[5] = rhs
    expected = DataFrame({'x': [1, 2, 9], 'y': [2.0, 2.0, 99.0]}, index=index)
    tm.assert_frame_equal(df, expected)