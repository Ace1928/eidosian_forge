from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
def test_fuse_slice():
    assert fuse_slice(slice(10, 15), slice(0, 5, 2)) == slice(10, 15, 2)
    assert fuse_slice((slice(100, 200),), (None, slice(10, 20))) == (None, slice(110, 120))
    assert fuse_slice((slice(100, 200),), (slice(10, 20), None)) == (slice(110, 120), None)
    assert fuse_slice((1,), (None,)) == (1, None)
    assert fuse_slice((1, slice(10, 20)), (None, None, 3, None)) == (1, None, None, 13, None)
    with pytest.raises(NotImplementedError):
        fuse_slice(slice(10, 15, 2), -1)
    with pytest.raises(NotImplementedError):
        fuse_slice(None, np.array([0, 0]))