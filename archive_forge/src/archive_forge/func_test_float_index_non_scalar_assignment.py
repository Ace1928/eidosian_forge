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
def test_float_index_non_scalar_assignment(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[1.0, 2.0, 3.0])
    df.loc[df.index[:2]] = 1
    expected = DataFrame({'a': [1, 1, 3], 'b': [1, 1, 5]}, index=df.index)
    tm.assert_frame_equal(expected, df)