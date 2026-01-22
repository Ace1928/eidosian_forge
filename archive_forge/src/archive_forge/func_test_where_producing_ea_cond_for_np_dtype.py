from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_producing_ea_cond_for_np_dtype():
    df = DataFrame({'a': Series([1, pd.NA, 2], dtype='Int64'), 'b': [1, 2, 3]})
    result = df.where(lambda x: x.apply(lambda y: y > 1, axis=1))
    expected = DataFrame({'a': Series([pd.NA, pd.NA, 2], dtype='Int64'), 'b': [np.nan, 2, 3]})
    tm.assert_frame_equal(result, expected)