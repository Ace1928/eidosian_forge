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
def test_inf_upcast(self):
    df = DataFrame(columns=[0])
    df.loc[1] = 1
    df.loc[2] = 2
    df.loc[np.inf] = 3
    assert df.loc[np.inf, 0] == 3
    result = df.index
    expected = Index([1, 2, np.inf], dtype=np.float64)
    tm.assert_index_equal(result, expected)