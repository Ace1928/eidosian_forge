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
def test_dont_fuse_numpy_arrays():
    x = np.ones(10)
    for _ in [(5,), (10,)]:
        y = da.from_array(x, chunks=(10,))
        dsk = y.__dask_optimize__(y.dask, y.__dask_keys__())
        assert sum((isinstance(v, np.ndarray) for v in dsk.values())) == 1