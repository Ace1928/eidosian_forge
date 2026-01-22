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
@pytest.mark.parametrize('max_seq_items, expected', [(5, '{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}'), (4, '{0: [0], 1: [1], 2: [2], 3: [3], ...}'), (1, '{0: [0], ...}')])
def test_groups_repr_truncates(max_seq_items, expected):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 1)))
    df['a'] = df.index
    with pd.option_context('display.max_seq_items', max_seq_items):
        result = df.groupby('a').groups.__repr__()
        assert result == expected
        result = df.groupby(np.array(df.a)).groups.__repr__()
        assert result == expected