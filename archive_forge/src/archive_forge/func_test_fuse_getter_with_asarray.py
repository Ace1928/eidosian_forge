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
@pytest.mark.parametrize('chunks', [10, 5, 3])
def test_fuse_getter_with_asarray(chunks):
    x = np.ones(10) * 1234567890
    y = da.ones(10, chunks=chunks)
    z = x + y
    dsk = z.__dask_optimize__(z.dask, z.__dask_keys__())
    assert any((v is x for v in dsk.values()))
    for v in dsk.values():
        s = str(v)
        assert s.count('getitem') + s.count('getter') <= 1
        if v is not x:
            assert '1234567890' not in s
    n_getters = len([v for v in dsk.values() if _is_getter_task(v)])
    if y.npartitions > 1:
        assert n_getters == y.npartitions
    else:
        assert n_getters == 0
    assert_eq(z, x + 1)