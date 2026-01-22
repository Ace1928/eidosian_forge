import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('obj', [Index([1, 2, 3]), Index([True, False, True]), timedelta_range('1 day', periods=2), period_range('2020-01-01', freq='D', periods=2), MultiIndex.from_product([range(5), ['foo', 'bar', 'baz'], pd.date_range('20130101', periods=2)]), MultiIndex.from_product([pd.CategoricalIndex(list('aabc')), range(3)])])
def test_hash_pandas_index(obj, index):
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)