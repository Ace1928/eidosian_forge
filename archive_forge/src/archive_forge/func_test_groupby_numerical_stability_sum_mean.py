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
@pytest.mark.parametrize('func, values', [('sum', [97.0, 98.0]), ('mean', [24.25, 24.5])])
def test_groupby_numerical_stability_sum_mean(func, values):
    data = [1e+16, 1e+16, 97, 98, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0]
    df = DataFrame({'group': [1, 2] * 4, 'a': data, 'b': data})
    result = getattr(df.groupby('group'), func)()
    expected = DataFrame({'a': values, 'b': values}, index=Index([1, 2], name='group'))
    tm.assert_frame_equal(result, expected)