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
def test_gh3937():
    x = da.from_array([1, 2, 3.0], (2,))
    x = da.concatenate((x, [x[-1]]))
    y = x.rechunk((2,))
    y = da.coarsen(np.sum, y, {0: 2})
    y.compute()