from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_update_inplace_sets_valid_block_values(using_copy_on_write):
    df = DataFrame({'a': Series([1, 2, None], dtype='category')})
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['a'].fillna(1, inplace=True)
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            df['a'].fillna(1, inplace=True)
    assert isinstance(df._mgr.blocks[0].values, Categorical)
    if not using_copy_on_write:
        assert df.isnull().sum().sum() == 0