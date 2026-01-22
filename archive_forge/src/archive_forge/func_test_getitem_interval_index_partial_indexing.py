from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_interval_index_partial_indexing(self):
    df = DataFrame(np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5)))
    expected = df.iloc[:, 0]
    res = df[0.5]
    tm.assert_series_equal(res, expected)
    res = df.loc[:, 0.5]
    tm.assert_series_equal(res, expected)