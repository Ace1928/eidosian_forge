import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_astype_object_frame(self, all_data):
    df = pd.DataFrame({'A': all_data})
    result = df.astype(object)
    if hasattr(result._mgr, 'blocks'):
        blk = result._mgr.blocks[0]
        assert isinstance(blk, NumpyBlock), type(blk)
        assert blk.is_object
    assert isinstance(result._mgr.arrays[0], np.ndarray)
    assert result._mgr.arrays[0].dtype == np.dtype(object)
    comp = result.dtypes == df.dtypes
    assert not comp.any()