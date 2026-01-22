import numpy as np
from pandas import (
import pandas._testing as tm
def test_np_sqrt(self, float_frame):
    with np.errstate(all='ignore'):
        result = np.sqrt(float_frame)
    assert isinstance(result, type(float_frame))
    assert result.index.is_(float_frame.index)
    assert result.columns.is_(float_frame.columns)
    tm.assert_frame_equal(result, float_frame.apply(np.sqrt))