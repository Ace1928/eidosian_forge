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
@pytest.mark.parametrize('level_arg, multiindex', [([0], False), ((0,), False), ([0], True), ((0,), True)])
def test_single_element_listlike_level_grouping_deprecation(level_arg, multiindex):
    df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, index=['x', 'y'])
    if multiindex:
        df = df.set_index(['a', 'b'])
    depr_msg = 'Creating a Groupby object with a length-1 list-like level parameter will yield indexes as tuples in a future version. To keep indexes as scalars, create Groupby objects with a scalar level parameter instead.'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        [key for key, _ in df.groupby(level=level_arg)]