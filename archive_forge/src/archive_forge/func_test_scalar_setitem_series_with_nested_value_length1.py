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
def test_scalar_setitem_series_with_nested_value_length1(value, indexer_sli):
    ser = Series([1.0, 2.0, 3.0])
    if isinstance(value, np.ndarray):
        indexer_sli(ser)[0] = value
        expected = Series([0.0, 2.0, 3.0])
        tm.assert_series_equal(ser, expected)
    else:
        with pytest.raises(ValueError, match='setting an array element with a sequence'):
            indexer_sli(ser)[0] = value
    ser = Series([1, 'a', 'b'], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value