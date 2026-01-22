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
def test_fuse_getitem_lock(getter, getter_nofancy, getitem):
    lock1 = SerializableLock()
    lock2 = SerializableLock()
    pairs = [((getter, (getter, 'x', slice(1000, 2000), True, lock1), slice(15, 20)), (getter, 'x', slice(1015, 1020), True, lock1)), ((getitem, (getter, 'x', (slice(1000, 2000), slice(100, 200)), True, lock1), (slice(15, 20), slice(50, 60))), (getter, 'x', (slice(1015, 1020), slice(150, 160)), True, lock1)), ((getitem, (getter_nofancy, 'x', (slice(1000, 2000), slice(100, 200)), True, lock1), (slice(15, 20), slice(50, 60))), (getter_nofancy, 'x', (slice(1015, 1020), slice(150, 160)), True, lock1)), ((getter, (getter, 'x', slice(1000, 2000), True, lock1), slice(15, 20), True, lock2), (getter, (getter, 'x', slice(1000, 2000), True, lock1), slice(15, 20), True, lock2))]
    for inp, expected in pairs:
        result = optimize_slices({'y': inp})
        _assert_getter_dsk_eq(result, {'y': expected})