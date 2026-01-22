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
def test_subsetting_columns_axis_1():
    df = DataFrame({'A': [1], 'B': [2], 'C': [3]})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g = df.groupby([0, 0, 1], axis=1)
    match = 'Cannot subset columns when using axis=1'
    with pytest.raises(ValueError, match=match):
        g[['A', 'B']].sum()