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
def test_groupby_ngroup_with_nan():
    df = DataFrame({'a': Categorical([np.nan]), 'b': [1]})
    result = df.groupby(['a', 'b'], dropna=False, observed=False).ngroup()
    expected = Series([0])
    tm.assert_series_equal(result, expected)