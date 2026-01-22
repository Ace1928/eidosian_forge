from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('index_dtype', [np.int64, np.float64])
def test_unsorted_index(self, index_dtype):
    df = DataFrame({'y': np.arange(100)}, index=Index(np.arange(99, -1, -1), dtype=index_dtype), dtype=np.int64)
    ax = df.plot()
    lines = ax.get_lines()[0]
    rs = lines.get_xydata()
    rs = Series(rs[:, 1], rs[:, 0], dtype=np.int64, name='y')
    tm.assert_series_equal(rs, df.y, check_index_type=False)