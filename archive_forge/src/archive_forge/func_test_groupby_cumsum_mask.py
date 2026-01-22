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
@pytest.mark.parametrize('skipna, val', [(True, 3), (False, pd.NA)])
def test_groupby_cumsum_mask(any_numeric_ea_dtype, skipna, val):
    df = DataFrame({'a': 1, 'b': [1, pd.NA, 2]}, dtype=any_numeric_ea_dtype)
    result = df.groupby('a').cumsum(skipna=skipna)
    expected = DataFrame({'b': [1, pd.NA, val]}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)