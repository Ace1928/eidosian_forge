from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_cython_grouper_series_bug_noncontig():
    arr = np.empty((100, 100))
    arr.fill(np.nan)
    obj = Series(arr[:, 0])
    inds = np.tile(range(10), 10)
    result = obj.groupby(inds).agg(Series.median)
    assert result.isna().all()