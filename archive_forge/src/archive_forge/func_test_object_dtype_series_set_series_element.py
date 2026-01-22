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
def test_object_dtype_series_set_series_element():
    s1 = Series(dtype='O', index=['a', 'b'])
    s1['a'] = Series()
    s1.loc['b'] = Series()
    tm.assert_series_equal(s1.loc['a'], Series())
    tm.assert_series_equal(s1.loc['b'], Series())
    s2 = Series(dtype='O', index=['a', 'b'])
    s2.iloc[1] = Series()
    tm.assert_series_equal(s2.iloc[1], Series())