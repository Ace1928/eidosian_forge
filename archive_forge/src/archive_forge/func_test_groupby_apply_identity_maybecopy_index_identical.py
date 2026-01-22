from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', [lambda x: x, lambda x: x[:], lambda x: x.copy(deep=False), lambda x: x.copy(deep=True)])
def test_groupby_apply_identity_maybecopy_index_identical(func):
    df = DataFrame({'g': [1, 2, 2, 2], 'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('g', group_keys=False).apply(func)
    tm.assert_frame_equal(result, df)