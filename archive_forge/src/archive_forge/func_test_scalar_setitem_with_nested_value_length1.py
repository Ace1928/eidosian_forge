import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.parametrize('value', [(0.0,), [0.0], np.array([0.0]), array.array('d', [0.0])])
def test_scalar_setitem_with_nested_value_length1(value):
    df = DataFrame({'A': [1, 2, 3]})
    df.loc[0, 'B'] = value
    expected = DataFrame({'A': [1, 2, 3], 'B': [0.0, np.nan, np.nan]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'A': [1, 2, 3], 'B': np.array([1, 'a', 'b'], dtype=object)})
    df.loc[0, 'B'] = value
    if isinstance(value, np.ndarray):
        assert (df.loc[0, 'B'] == value).all()
    else:
        assert df.loc[0, 'B'] == value