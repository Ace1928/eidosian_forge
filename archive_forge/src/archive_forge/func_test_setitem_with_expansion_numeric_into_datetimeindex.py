import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key', [100, 100.0])
def test_setitem_with_expansion_numeric_into_datetimeindex(self, key):
    orig = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df = orig.copy()
    df.loc[key, :] = df.iloc[0]
    ex_index = Index(list(orig.index) + [key], dtype=object, name=orig.index.name)
    ex_data = np.concatenate([orig.values, df.iloc[[0]].values], axis=0)
    expected = DataFrame(ex_data, index=ex_index, columns=orig.columns)
    tm.assert_frame_equal(df, expected)