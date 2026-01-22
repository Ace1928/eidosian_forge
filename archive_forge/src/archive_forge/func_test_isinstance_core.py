from __future__ import annotations
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import pytest
import dask
import dask.threaded
from dask.base import DaskMethodsMixin, dont_optimize, tokenize
from dask.context import globalmethod
from dask.delayed import Delayed, delayed
from dask.typing import (
@pytest.mark.parametrize('protocol', [DaskCollection, HLGDaskCollection])
def test_isinstance_core(protocol):
    arr = da.ones(10)
    bag = db.from_sequence([1, 2, 3, 4, 5], npartitions=2)
    df = dds.timeseries()
    dobj = increment(2)
    assert_isinstance(arr, protocol)
    assert_isinstance(bag, protocol)
    if not dd._dask_expr_enabled():
        assert_isinstance(df, protocol)
    assert_isinstance(dobj, protocol)