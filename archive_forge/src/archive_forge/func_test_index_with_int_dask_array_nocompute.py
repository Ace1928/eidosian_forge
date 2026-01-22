from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_index_with_int_dask_array_nocompute():
    """Test that when the indices are a dask array
    they are not accidentally computed
    """

    def crash():
        raise NotImplementedError()
    x = da.arange(5, chunks=-1)
    idx = da.Array({('x', 0): (crash,)}, name='x', chunks=((2,),), dtype=np.int64)
    result = x[idx]
    with pytest.raises(NotImplementedError):
        result.compute()