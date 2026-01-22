from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_squeeze_axis(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=1, freq='B')).iloc[:, :1]
    assert df.shape == (1, 1)
    tm.assert_series_equal(df.squeeze(axis=0), df.iloc[0])
    tm.assert_series_equal(df.squeeze(axis='index'), df.iloc[0])
    tm.assert_series_equal(df.squeeze(axis=1), df.iloc[:, 0])
    tm.assert_series_equal(df.squeeze(axis='columns'), df.iloc[:, 0])
    assert df.squeeze() == df.iloc[0, 0]
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        df.squeeze(axis=2)
    msg = 'No axis named x for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        df.squeeze(axis='x')