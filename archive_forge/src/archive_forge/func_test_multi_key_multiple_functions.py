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
def test_multi_key_multiple_functions(df):
    grouped = df.groupby(['A', 'B'])['C']
    agged = grouped.agg(['mean', 'std'])
    expected = DataFrame({'mean': grouped.agg('mean'), 'std': grouped.agg('std')})
    tm.assert_frame_equal(agged, expected)