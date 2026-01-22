import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [slice(2, 4.0), slice(2.0, 4), slice(2.0, 4.0)])
def test_integer_positional_indexing(self, idx):
    """make sure that we are raising on positional indexing
        w.r.t. an integer index
        """
    s = Series(range(2, 6), index=range(2, 6))
    result = s[2:4]
    expected = s.iloc[2:4]
    tm.assert_series_equal(result, expected)
    klass = RangeIndex
    msg = f'cannot do (slice|positional) indexing on {klass.__name__} with these indexers \\[(2|4)\\.0\\] of type float'
    with pytest.raises(TypeError, match=msg):
        s[idx]
    with pytest.raises(TypeError, match=msg):
        s.iloc[idx]