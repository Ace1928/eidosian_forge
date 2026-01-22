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
@pytest.mark.filterwarnings('ignore:invalid value encountered in remainder:RuntimeWarning')
@pytest.mark.parametrize('method', ['head', 'tail', 'nth', 'first', 'last'])
def test_groupby_method_drop_na(method):
    df = DataFrame({'A': ['a', np.nan, 'b', np.nan, 'c'], 'B': range(5)})
    if method == 'nth':
        result = getattr(df.groupby('A'), method)(n=0)
    else:
        result = getattr(df.groupby('A'), method)()
    if method in ['first', 'last']:
        expected = DataFrame({'B': [0, 2, 4]}).set_index(Series(['a', 'b', 'c'], name='A'))
    else:
        expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [0, 2, 4]}, index=[0, 2, 4])
    tm.assert_frame_equal(result, expected)