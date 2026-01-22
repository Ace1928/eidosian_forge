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
def test_iloc_getitem_indexing_dtypes_on_empty(self):
    df = DataFrame({'a': [1, 2, 3], 'b': ['b', 'b2', 'b3']})
    df2 = df.iloc[[], :]
    assert df2.loc[:, 'a'].dtype == np.int64
    tm.assert_series_equal(df2.loc[:, 'a'], df2.iloc[:, 0])