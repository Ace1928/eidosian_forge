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
def test_group_on_two_row_multiindex_returns_one_tuple_key():
    df = DataFrame([{'a': 1, 'b': 2, 'c': 99}, {'a': 1, 'b': 2, 'c': 88}])
    df = df.set_index(['a', 'b'])
    grp = df.groupby(['a', 'b'])
    result = grp.indices
    expected = {(1, 2): np.array([0, 1], dtype=np.int64)}
    assert len(result) == 1
    key = (1, 2)
    assert (result[key] == expected[key]).all()