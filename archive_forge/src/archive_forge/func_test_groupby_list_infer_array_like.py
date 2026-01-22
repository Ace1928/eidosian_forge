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
def test_groupby_list_infer_array_like(df):
    result = df.groupby(list(df['A'])).mean(numeric_only=True)
    expected = df.groupby(df['A']).mean(numeric_only=True)
    tm.assert_frame_equal(result, expected, check_names=False)
    with pytest.raises(KeyError, match="^'foo'$"):
        df.groupby(list(df['A'][:-1]))
    df = DataFrame({'foo': [0, 1], 'bar': [3, 4], 'val': np.random.default_rng(2).standard_normal(2)})
    result = df.groupby(['foo', 'bar']).mean()
    expected = df.groupby([df['foo'], df['bar']]).mean()[['val']]