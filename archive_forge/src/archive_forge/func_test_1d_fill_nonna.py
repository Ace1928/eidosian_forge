from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_1d_fill_nonna(self, dtype_fill_out_dtype):
    dtype, fill_value, out_dtype = dtype_fill_out_dtype
    data = np.random.default_rng(2).integers(0, 2, 4).astype(dtype)
    indexer = [2, 1, 0, -1]
    result = algos.take_nd(data, indexer, fill_value=fill_value)
    assert (result[[0, 1, 2]] == data[[2, 1, 0]]).all()
    assert result[3] == fill_value
    assert result.dtype == out_dtype
    indexer = [2, 1, 0, 1]
    result = algos.take_nd(data, indexer, fill_value=fill_value)
    assert (result[[0, 1, 2, 3]] == data[indexer]).all()
    assert result.dtype == dtype