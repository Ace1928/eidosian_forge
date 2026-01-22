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
@pytest.mark.parametrize('agg_function', ['max', 'min'])
def test_keep_nuisance_agg(df, agg_function):
    grouped = df.groupby('A')
    result = getattr(grouped, agg_function)()
    expected = result.copy()
    expected.loc['bar', 'B'] = getattr(df.loc[df['A'] == 'bar', 'B'], agg_function)()
    expected.loc['foo', 'B'] = getattr(df.loc[df['A'] == 'foo', 'B'], agg_function)()
    tm.assert_frame_equal(result, expected)