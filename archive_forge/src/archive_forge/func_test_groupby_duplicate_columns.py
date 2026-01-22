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
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_groupby_duplicate_columns(infer_string):
    if infer_string:
        pytest.importorskip('pyarrow')
    df = DataFrame({'A': ['f', 'e', 'g', 'h'], 'B': ['a', 'b', 'c', 'd'], 'C': [1, 2, 3, 4]}).astype(object)
    df.columns = ['A', 'B', 'B']
    with pd.option_context('future.infer_string', infer_string):
        result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame([['e', 'a', 1]], index=np.array([0]), columns=['A', 'B', 'B'], dtype=object)
    tm.assert_frame_equal(result, expected)