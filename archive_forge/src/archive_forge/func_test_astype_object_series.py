import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_astype_object_series(self, all_data):
    ser = pd.Series(all_data, name='A')
    result = ser.astype(object)
    assert result.dtype == np.dtype(object)
    if hasattr(result._mgr, 'blocks'):
        blk = result._mgr.blocks[0]
        assert isinstance(blk, NumpyBlock)
        assert blk.is_object
    assert isinstance(result._mgr.array, np.ndarray)
    assert result._mgr.array.dtype == np.dtype(object)