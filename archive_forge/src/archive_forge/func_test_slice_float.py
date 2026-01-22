import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
def test_slice_float(self, idx, frame_or_series, indexer_sl):
    index = Index(np.arange(5.0)) + 0.1
    s = gen_obj(frame_or_series, index)
    expected = s.iloc[3:4]
    result = indexer_sl(s)[idx]
    assert isinstance(result, type(s))
    tm.assert_equal(result, expected)
    s2 = s.copy()
    indexer_sl(s2)[idx] = 0
    result = indexer_sl(s2)[idx].values.ravel()
    assert (result == 0).all()